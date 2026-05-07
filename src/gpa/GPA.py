
import math
from math import sqrt, pow, sin, cos, atan2, pi, fabs
from scipy.spatial import Delaunay
from scipy.spatial import QhullError

import importlib.util
hasCupy = importlib.util.find_spec("cupyx")
if hasCupy:
  import cupyx as np
  #from cupyx.scipy.spatial import cKDTree
  from cupyx.scipy.spatial import KDTree as cKDTree
  import numpy as np
  import cupy as cp
else:
  import numpy as np
  from scipy.spatial import cKDTree


class GPA:
    def __init__(self, tol=0.03):
        self.mat = None
        self.gradient_dx = None
        self.gradient_dy = None
        self.cx = 0.0
        self.cy = 0.0
        self.rows = 0
        self.cols = 0

        self.phases = None
        self.mods = None
        self.symmetricalP = np.array([[]], dtype=np.int32)
        self.asymmetricalP = np.array([[]], dtype=np.int32)
        self.unknownP = np.array([[]], dtype=np.int32)
        self.triangulation_points = []
        self.triangles = None

        self.maxGrad = 0.0
        self.tol = tol
        self.cvet = None

        self.n_edges = 0
        self.n_points = 0
        self.G1 = 0.0
        self.G2 = 0.0
        self.G3 = 0.0
        self.G1_Classic = 0.0
        self.G4 = None

    def setPosition(self, cx, cy):
        self.cx = cx
        self.cy = cy

    def _setGradients(self):
        gy, gx = np.gradient(self.mat)

        self.gradient_dx = gx
        self.gradient_dy = gy

        self._setMaxGrad()
        self._setModulusPhase()

    def _setMaxGrad(self):
        self.maxGrad = np.max(np.hypot(self.gradient_dy, self.gradient_dx))
        if self.maxGrad < 1e-5:
            self.maxGrad = 1.0

    def _setModulusPhase(self):
        w, h = self.cols, self.rows
        gx = self.gradient_dx
        gy = self.gradient_dy

        # Cálculo de fases usando list comprehension aninhada (preservado do original)
        phases_list = []
        for j in range(h):
            row = []
            for i in range(w):
                val = atan2(gy[j, i], gx[j, i])
                if val <= 0:
                    row.append(val + 2.0 * pi)
                else:
                    row.append(val)
            phases_list.append(row)
        self.phases = np.array(phases_list, dtype=np.float64)

        # Cálculo de módulos
        mods_list = []
        for j in range(h):
            row = []
            for i in range(w):
                row.append(self.getMod(gx[j, i], gy[j, i]))
            mods_list.append(row)
        self.mods = np.array(mods_list, dtype=np.float64)

    def getMod(self, x, y):
        return sqrt(pow(x, 2.0) + pow(y, 2.0)) / self.maxGrad

    def version(self):
        return "GPA - 3.6"

    def _update_asymmetric_mat(self, index_dist, dists, tol, ptol):
        # Preparação das estruturas de dados (Vetorização Global)
        rows, cols = self.rows, self.cols
        self.symmetricalP = np.zeros((rows, cols), dtype=int)
        self.asymmetricalP = np.zeros((rows, cols), dtype=int)
        self.unknownP = np.zeros((rows, cols), dtype=int)

        # 1. Definir Unknowns globalmente
        low_mod_mask = self.mods <= tol
        self.unknownP[low_mod_mask] = 1

        # Achatamento para 1D (facilita indexação)
        flat_dx = self.gradient_dx.ravel()
        flat_dy = self.gradient_dy.ravel()
        flat_dists = dists.ravel()
        flat_unknown = self.unknownP.ravel()
        flat_indices = np.arange(rows * cols)

        # Limiar real de distância (tolerância * maxGrad)
        # Como vamos comparar dist(v1, -v2), não elevamos ao quadrado aqui para usar a saída direta da KDTree
        real_tol = tol * self.maxGrad

        # 2. Loop pelos anéis de distância
        for d in index_dist:
            # Filtra pontos no anel atual que não são 'unknown'
            # abs(dists - d) <= ptol
            on_ring_mask = np.abs(flat_dists - d) <= abs(ptol)
            valid_mask = on_ring_mask & (flat_unknown == 0)

            # Índices e vetores ativos
            current_indices = flat_indices[valid_mask]
            n_points = len(current_indices)

            # Se não houver par possível, pule
            if n_points < 2:
                continue

            # Monta matriz de vetores (N, 2)
            # v = [dx, dy]
            vectors = np.column_stack((flat_dx[current_indices], flat_dy[current_indices]))
            if hasCupy:
              vectors = cp.asarray(vectors)


            # --- OTIMIZAÇÃO VIA KDTREE (N log N) ---
            # Em vez de comparar todos com todos, montamos uma árvore espacial
            tree = cKDTree(vectors)

            # Procuramos pelo vetor OPOSTO (-dx, -dy)
            # k=1 retorna apenas o vizinho mais próximo
            # workers=-1 usa todos os núcleos da CPU disponíveis
            if hasCupy:
              distances, _ = tree.query(-vectors, k=1)
            else:
              distances, _ = tree.query(-vectors, k=1, workers=-1)

            # Se a distância entre o vetor v e o vetor oposto mais próximo for pequena,
            # então v + vizinho ≈ 0. Existe simetria.
            # A comparação direta evita matrizes quadráticas gigantes.
            matches = distances <= real_tol

            # Mapeia de volta para a matriz da imagem
            if np.any(matches):
                sym_indices_linear = current_indices[matches.get()]
                ys, xs = np.unravel_index(sym_indices_linear, (rows, cols))
                self.symmetricalP[ys, xs] = 1

        # 3. Definir Assimétricos (Bitwise logic)
        mask_sym = self.symmetricalP == 1
        mask_unk = self.unknownP == 1
        # O que não é Simétrico nem Unknown, é Assimétrico
        self.asymmetricalP[~(mask_sym | mask_unk)] = 1

    def _G1_Classic(self, symm):
        targetMat = None
        self.triangulation_points = []

        if symm == 'S':
            targetMat = self.symmetricalP
        elif symm == 'A':
            targetMat = self.asymmetricalP
        elif symm == 'F':
            targetMat = np.ones((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
        else:
            raise Exception("Unknown analysis type (should be S,A or K), got: " + symm)

        for i in range(self.rows):
            for j in range(self.cols):
                if targetMat[i, j] > 0:
                    self.triangulation_points.append([j + 0.5 * self.gradient_dx[i, j], i + 0.5 * self.gradient_dy[i, j]])

        # Conversão e verificação de pontos únicos
        tp_array = np.array(self.triangulation_points)
        if len(tp_array) > 1:
            try:
                self.triangulation_points = np.unique(tp_array, axis=0)
            except TypeError:
                # Fallback para versões antigas ou compatibilidade
                b = np.ascontiguousarray(tp_array).view(np.dtype((np.void, tp_array.dtype.itemsize * tp_array.shape[1])))
                _, idx = np.unique(b, return_index=True)
                self.triangulation_points = tp_array[idx]
        else:
            self.triangulation_points = tp_array

        self.n_points = len(self.triangulation_points)

        if self.n_points < 3:
            self.n_edges = 0
            self.G1_Classic = 0.0
        else:
            try:
                self.triangles = Delaunay(self.triangulation_points)
                neigh = self.triangles.vertex_neighbor_vertices
                self.n_edges = len(neigh[1]) / 2
                self.G1_Classic = (float(self.n_edges) - float(self.n_points)) / float(self.n_points)
            except QhullError:
                self.n_edges = 0
                self.G1_Classic = 0.0
                self.n_points = 0

        if self.G1_Classic < 0.0:
            self.G1_Classic = 0.0

    def _getDistancesTriang(self, points, simplices):
        ds = []
        for p in simplices:
            p1 = points[p[0]]
            p2 = points[p[1]]
            p3 = points[p[2]]
            ds.append(np.sqrt(np.sum((p1 - p2)**2)))
            ds.append(np.sqrt(np.sum((p2 - p3)**2)))
            ds.append(np.sqrt(np.sum((p3 - p1)**2)))
        return ds

    def _G1(self, symm):
        targetMat = None
        self.triangulation_points = []

        if symm == 'S':
            targetMat = self.symmetricalP
        elif symm == 'A':
            targetMat = self.asymmetricalP
        elif symm == 'F':
            targetMat = np.ones((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
        else:
            raise Exception("Unknown analysis type (should be S,A or K), got: " + symm)

        for i in range(self.rows):
            for j in range(self.cols):
                if targetMat[i, j] > 0:
                    self.triangulation_points.append([j + 0.5 * self.gradient_dx[i, j], i + 0.5 * self.gradient_dy[i, j]])

        tp_array = np.array(self.triangulation_points)
        if len(tp_array) > 1:
            try:
                self.triangulation_points = np.unique(tp_array, axis=0)
            except TypeError:
                b = np.ascontiguousarray(tp_array).view(np.dtype((np.void, tp_array.dtype.itemsize * tp_array.shape[1])))
                _, idx = np.unique(b, return_index=True)
                self.triangulation_points = tp_array[idx]
        else:
            self.triangulation_points = tp_array

        self.n_points = len(self.triangulation_points)

        if self.n_points < 3:
            self.n_edges = 0
            self.G1 = 0.0
        else:
            try:
                self.triangles = Delaunay(self.triangulation_points)
                neigh = self.triangles.vertex_neighbor_vertices
                self.n_edges = len(neigh[1]) / 2
                ds = self._getDistancesTriang(self.triangulation_points, self.triangles.simplices)
                ds = np.sort(ds) / np.max(ds)
                self.G1 = (np.average(ds[len(ds)//2:]) - np.average(ds[:len(ds)//2])) / np.max(ds)
            except QhullError:
                self.n_edges = 0
                self.G1 = 0.0
                self.n_points = 0

        if self.G1 < 0.0:
            self.G1 = 0.0

    def _G2(self, symm):
        targetMat = None
        opositeMat = None
        probabilityMat = None
        somax = 0.0
        somay = 0.0
        smod = 0.0

        if symm == 'S':
            targetMat = self.symmetricalP
            opositeMat = self.asymmetricalP
        elif symm == 'A':
            targetMat = self.asymmetricalP
            opositeMat = self.symmetricalP
        elif symm == 'F':
            targetMat = np.ones((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
            opositeMat = np.zeros((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
        else:
            raise Exception("Unknown analysis type (should be S,A or F), got: " + symm)

        if np.sum(targetMat) < 1:
            self.G2 = 0.0
            return

        alinhamento = 0.0

        if symm != 'S':
            for i in range(self.rows):
                for j in range(self.cols):
                    if targetMat[i, j] == 1:
                        somax += self.gradient_dx[i, j] / self.maxGrad
                        somay += self.gradient_dy[i, j] / self.maxGrad
                        smod += self.mods[i, j]
            if smod <= 0.0:
                alinhamento = 0.0
            else:
                alinhamento = sqrt(pow(somax, 2.0) + pow(somay, 2.0)) / (2 * smod)

            total_sum = np.sum(opositeMat) + np.sum(targetMat)
            if total_sum > 0:
                self.G2 = (float(np.sum(targetMat)) / float(total_sum)) * (1.0 - alinhamento)
            else:
                self.G2 = 0.0
        else:
            probabilityMat = self.mods * np.array(targetMat, dtype=np.float64)
            prob_sum = np.sum(probabilityMat)
            if prob_sum > 0:
                probabilityMat = probabilityMat / prob_sum

            sum_target = np.sum(targetMat)
            if sum_target > 0:
                maxEntropy = np.log(np.float64(sum_target))

                for i in range(self.rows):
                    for j in range(self.cols):
                        if targetMat[i, j] == 1 and probabilityMat[i, j] > 0:
                            alinhamento = alinhamento - probabilityMat[i, j] * np.log(probabilityMat[i, j]) / maxEntropy
                self.G2 = alinhamento
            else:
                self.G2 = 0.0

    def distAngle(self, a1, a2):
        return (cos(a1) * cos(a2) + sin(a1) * sin(a2) + 1) / 2

    def _G3(self, symm):
        targetMat = None
        opositeMat = None

        if symm == 'S':
            targetMat = self.symmetricalP
            opositeMat = self.asymmetricalP
        elif symm == 'A':
            targetMat = self.asymmetricalP
            opositeMat = self.symmetricalP
        elif symm == 'F':
            targetMat = np.ones((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
            opositeMat = np.zeros((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
        else:
            raise Exception("Unknown analysis type (should be S,A or F), got: " + symm)

        # Coletar índices onde targetMat > 0
        targetList = []
        for ty in range(self.rows):
            for tx in range(self.cols):
                if targetMat[ty, tx] > 0:
                    targetList.append([ty, tx])

        sumPhases = 0.0
        nterms = 0.0
        alinhamento = 0.0

        for coord in targetList:
            x1, y1 = coord[0], coord[1]
            y2, x2 = x1 - int(self.cx), y1 - int(self.cy)

            val = atan2(y2, x2)
            angle = val if val > 0 else val + 2.0 * pi

            sumPhases += self.distAngle(self.phases[x1, y1], angle)
            nterms += 1.0

        if nterms > 0.0:
            alinhamento = sumPhases / nterms
        else:
            alinhamento = 0.0

        total_sum = np.sum(opositeMat) + np.sum(targetMat)
        if total_sum > 0:
            self.G3 = ((float(np.sum(targetMat)) / float(total_sum)) + alinhamento) / 2
        else:
            self.G3 = 0.0

    def _G4(self, symm):
        targetMat = None
        sumZ = 0.0

        if symm == 'S':
            targetMat = self.symmetricalP
        elif symm == 'A':
            targetMat = self.asymmetricalP
        elif symm == 'F':
            targetMat = np.ones((self.symmetricalP.shape[0], self.symmetricalP.shape[1]), dtype=np.int32)
        else:
            raise Exception("Unknown analysis type (should be S,A or F ), got: " + symm)

        self.G4 = 0.0 + 0.0j
        sumZ = 0.0

        # Primeiro loop para calcular sumZ
        for i in range(self.rows):
            for j in range(self.cols):
                if targetMat[i, j] > 0:
                    if self.mods[i, j] > 1e-6:
                        sumZ += np.abs(self.mods[i, j] * np.exp(1j * self.phases[i, j]))

        if sumZ < 1e-5:
            return self.G4

        # Segundo loop para calcular entropia complexa
        for i in range(self.rows):
            for j in range(self.cols):
                if targetMat[i, j] > 0:
                    if self.mods[i, j] > 1e-6:
                        z = self.mods[i, j] * np.exp(1j * self.phases[i, j]) / sumZ
                        self.G4 = self.G4 - z * np.log(z)
        return self.G4

    def __call__(self, mat=None, gx=None, gy=None, moment=["G2"], symmetrycalGrad='A',precision=3):
        if (mat is None) and (gx is None) and (gy is None):
            raise Exception("Matrix or gradient must be stated!")
        if ((gx is None) and not (gy is None)) or (not (gx is None) and (gy is None)):
            raise Exception("Gradient must have 2 components (gx and gy)")
        if not (mat is None) and not (gx is None):
            raise Exception("Matrix or gradient must be stated, not both")

        if not (mat is None):
            return self._eval(mat, moment, symmetrycalGrad,precision)
        else:
            return self._evalGradient(gx, gy, moment, symmetrycalGrad)

    def _eval(self, mat, moment=["G2"], symmetrycalGrad='A',precision=3):
        self.mat = mat
        self.cols = len(self.mat[0])
        self.rows = len(self.mat)
        self.setPosition(float(self.rows - 1) / 2.0, float(self.cols - 1) / 2.0)
        self._setGradients()

        # Cálculo de distâncias
        dists_list = []
        for y in range(self.rows):
            row = []
            for x in range(self.cols):
                row.append(sqrt(pow(float(x) - self.cx, 2.0) + pow(float(y) - self.cy, 2.0)))
            dists_list.append(row)
        dists = np.array(dists_list, dtype=np.float64)

        minimo, maximo = np.min(dists), np.max(dists)
        sequence = np.arange(minimo, maximo, 0.705).astype(dtype=np.float64)
        uniq = np.array([m for m in sequence])

        # Remove simetria
        self._update_asymmetric_mat(uniq.astype(dtype=np.float64), dists.astype(dtype=np.float64), self.tol, 1.41)

        # Momentos do gradiente
        retorno = {}
        for gmoment in moment:
            if "G4" == gmoment:
                self._G4(symmetrycalGrad)
                retorno["G4"] = np.round(self.G4,precision)
            if "G3" == gmoment:
                self._G3(symmetrycalGrad)
                retorno["G3"] = np.round(self.G3,precision)
            if "G2" == gmoment:
                self._G2(symmetrycalGrad)
                retorno["G2"] = np.round(self.G2,precision)
            if "G1" == gmoment:
                self._G1(symmetrycalGrad)
                retorno["G1"] = np.round(self.G1,precision)
            if "G1C" == gmoment:
                self._G1_Classic(symmetrycalGrad)
                retorno["G1C"] = np.round(self.G1_Classic,precision)
        return retorno

    def getAsymmetricalMask(self):
        return np.array(self.asymmetricalP)

    def getSymmetricalMask(self):
        return np.array(self.symmetricalP)

    def getUnknownMask(self):
        return np.array(self.unknownP)

    def getDx(self):
        return np.array(self.gradient_dx)

    def getDy(self):
        return np.array(self.gradient_dy)

    def _evalGradient(self, gradient_dx, gradient_dy, moment=["G2"], symmetrycalGrad='A'):
        self.cols = len(gradient_dx[0])
        self.rows = len(gradient_dx)

        self.gradient_dx = gradient_dx
        self.gradient_dy = gradient_dy

        self._setMaxGrad()
        self._setModulusPhase()

        self.setPosition(float(self.rows - 1) / 2.0, float(self.cols - 1) / 2.0)

        dists_list = []
        for y in range(self.rows):
            row = []
            for x in range(self.cols):
                row.append(sqrt(pow(float(x) - self.cx, 2.0) + pow(float(y) - self.cy, 2.0)))
            dists_list.append(row)
        dists = np.array(dists_list, dtype=np.float64)

        minimo, maximo = np.min(dists), np.max(dists)
        sequence = np.arange(minimo, maximo, 0.705).astype(dtype=np.float64)
        uniq = np.array([m for m in sequence])

        self._update_asymmetric_mat(uniq.astype(dtype=np.float64), dists.astype(dtype=np.float64), self.tol, 1.41)

        retorno = {}
        for gmoment in moment:
            if "G4" == gmoment:
                self._G4(symmetrycalGrad)
                retorno["G4"] = self.G4
            if "G3" == gmoment:
                self._G3(symmetrycalGrad)
                retorno["G3"] = self.G3
            if "G2" == gmoment:
                self._G2(symmetrycalGrad)
                retorno["G2"] = self.G2
            if "G1" == gmoment:
                self._G1(symmetrycalGrad)
                retorno["G1"] = self.G1
            if "G1C" == gmoment:
                self._G1_Classic(symmetrycalGrad)
                retorno["G1C"] = self.G1_Classic
        return retorno
