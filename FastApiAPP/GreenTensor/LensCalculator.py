from Lens import Lens
import cmath
import scipy
import math
import numpy as np

class LensCalculator:
    def __init__(self, lens: Lens):
        """
        Инициализация калькулятора LensCalculator
        
        Parameters:
        -----------
        lens : Lens
            Объект линзы с параметрами
        """
        self.lens = lens
        self._initialize_arrays()
        
    def _initialize_arrays(self):
        """Инициализация всех массивов и переменных"""
        # Углы для расчета
        self.teta_start = 0.01
        self.teta_stop = 360
        self.step = math.pi/180
        self.teta_diap = abs(self.teta_stop) - abs(self.teta_start)
        self.steps = int(((self.teta_diap * (math.pi/180)) / self.step)) + 1
        self.teta = np.zeros(int(self.steps))
        self.cos_teta = np.zeros(int(self.steps))
        
        # Переменные среды
        self.alfa = np.zeros(self.lens.n, dtype=complex)
        self.beta = np.zeros(self.lens.n, dtype=complex)
        self.sigma = np.zeros(self.lens.n, dtype=complex)
        self.k = np.zeros((self.lens.n, self.lens.n), dtype=complex)
        
        # Функции Бесселя, Неймана и др.
        self.J = np.zeros(self.lens.toch, dtype=complex)
        self.Jpr = np.zeros(self.lens.toch, dtype=complex)
        self.N = np.zeros(self.lens.toch, dtype=complex)
        self.Npr = np.zeros(self.lens.toch, dtype=complex)
        self.C = np.zeros((self.lens.toch, self.lens.n-1), dtype=complex)
        self.Cpr = np.zeros((self.lens.toch, self.lens.n-1), dtype=complex)
        self.S = np.zeros((self.lens.toch, self.lens.n-1), dtype=complex)
        self.Spr = np.zeros((self.lens.toch, self.lens.n-1), dtype=complex)
        
        # Импедансы и адмитансы
        self.Z = np.zeros((self.lens.toch, len(self.lens.a)), dtype=complex)
        self.Y = np.zeros((self.lens.toch, len(self.lens.a)), dtype=complex)
        
        # Модифицированные функции
        self.mJ = np.zeros(self.lens.toch, dtype=complex)
        self.mJpr = np.zeros(self.lens.toch, dtype=complex)
        self.mH = np.zeros(self.lens.toch, dtype=complex)
        self.mHpr = np.zeros(self.lens.toch, dtype=complex)
        
        # Коэффициенты рассеяния
        self.Mn = np.zeros(self.lens.toch, dtype=complex)
        self.Nn = np.zeros(self.lens.toch, dtype=complex)
        
        # Поля
        self.E_kp = np.zeros((self.lens.toch, self.steps), dtype=complex)
        self.E_op = np.zeros((self.lens.toch, self.steps), dtype=complex)
        self.S_teta = np.zeros((self.lens.toch, self.steps), dtype=complex)
        self.S_phi = np.zeros((self.lens.toch, self.steps), dtype=complex)
        self.E_teta = np.zeros((1, self.steps), dtype=complex)
        self.E_phi = np.zeros((1, self.steps), dtype=complex)
        
    def calculate_medium_parameters(self):
        """Расчет параметров среды"""
        for i in range(self.lens.n):
            self.alfa[i] = cmath.atan(self.lens.eps[i].imag / self.lens.eps[i].real) if self.lens.eps[i].real != 0 else math.pi/2
            self.beta[i] = math.atan(self.lens.miy[i].imag / self.lens.miy[i].real)
            self.sigma[i] = cmath.sqrt(abs(self.lens.eps[i]) * abs(self.lens.miy[i]))

        if self.lens.eps[-1] != len(self.lens.eps) - 1:
            self.alfa = np.append(self.alfa, 0)
            self.lens.eps = np.append(self.lens.eps, len(self.lens.eps))
    
    def calculate_k_coefficients(self):
        """Расчет коэффициентов k"""
        j = 0
        for i in range(self.lens.n):
            self.k[i][j] = self.lens.k0 * self.lens.a[i] * self.sigma[j]
            if j < self.lens.n - 1:
                j += 1
                self.k[i][j] = self.lens.k0 * self.lens.a[i] * self.sigma[j]

    def Jfunc(self, i, j1, j2):
        """Функция Бесселя первого рода"""
        nu = i + 1
        J = (scipy.special.jv(nu + 0.5, self.k[j1][j2])) * (cmath.sqrt(self.k[j1][j2] * math.pi/2))
        return J
    
    def Jprfunc(self, i, j1, j2, tie):
        """Производная функции Бесселя первого рода"""
        nu = i + 1

        if tie == False:
            Jpr = ((nu / (2 * nu + 1)) *  (scipy.special.jv(nu - 0.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2)) - \
            ((nu + 1) / (2 * nu + 1)) *  (scipy.special.jv(nu + 1.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2)) + \
            (self.J[i] / self.k[j1][j2]))
        else:
            Jpr = ((nu / (2 * nu + 1)) * ((scipy.special.jv(nu - 0.5,self.k[j1][j2]) * (cmath.sqrt(self.k[j1][j2] * math.pi/2))) / self.k[j1][j2])) * self.k[j1][j2] - \
            (((nu + 1) / (2 * nu + 1)) * ((scipy.special.jv(nu + 1.5,self.k[j1][j2])) * (cmath.sqrt(self.k[j1][j2] * math.pi/2))) / self.k[j1][j2]) * self.k[j1][j2] + \
            ((scipy.special.jv(nu + 0.5,self.k[j1][j2])) * (cmath.sqrt(self.k[j1][j2] * math.pi/2))) / self.k[j1][j2]
        return Jpr
    
    def Nfunc(self, i, j1, j2):
        """Функция Неймана"""
        return scipy.special.yv((i+1) + 0.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2)
    
    def Nprfunc(self, i, j1, j2, tie):
        """Производная функции Неймана"""
        nu = i + 1
        if not tie:
            return ((nu / (2 * nu + 1)) * scipy.special.yv(nu - 0.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2) -
                   ((nu + 1) / (2 * nu + 1)) * scipy.special.yv(nu + 1.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2) +
                   (self.Nfunc(i, j1, j2) / self.k[j1][j2]))
        else:
            return ((nu / (2 * nu + 1)) * (scipy.special.yv(nu - 0.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2) / self.k[j1][j2]) * self.k[j1][j2] -
                   ((nu + 1) / (2 * nu + 1)) * (scipy.special.yv(nu + 1.5, self.k[j1][j2]) * cmath.sqrt(self.k[j1][j2] * math.pi/2) / self.k[j1][j2]) * self.k[j1][j2] +
                   (self.Nfunc(i, j1, j2) / self.k[j1][j2]))
    
    def calculate_bessel_functions(self):
        """Расчет функций Бесселя и Неймана"""
        for i in range(self.lens.toch):
            self.J[i] = self.Jfunc(i, 0, 0)
            self.Jpr[i] = self.Jprfunc(i, 0, 0, False)
            self.N[i] = self.Nfunc(i, 0, 0)
            self.Npr[i] = self.Nprfunc(i, 0, 0, False)
            
    def calculate_CS_functions(self):
        """Расчет функций C, S и их производных"""
        for i in range(self.lens.toch):
            for j in range(len(self.sigma)-1):
                self.C[i][j] = (self.Jfunc(i, j+1, j+1) * self.Nprfunc(i, j, j+1, True) - 
                               self.Nfunc(i, j+1, j+1) * self.Jprfunc(i, j, j+1, True))
                self.Cpr[i][j] = (self.Jprfunc(i, j+1, j+1, True) * self.Nprfunc(i, j, j+1, True) - 
                                 self.Nprfunc(i, j+1, j+1, True) * self.Jprfunc(i, j, j+1, True))
                self.S[i][j] = (self.Nfunc(i, j+1, j+1) * self.Jfunc(i, j, j+1) - 
                               self.Jfunc(i, j+1, j+1) * self.Nfunc(i, j, j+1))
                self.Spr[i][j] = (self.Nprfunc(i, j+1, j+1, True) * self.Jfunc(i, j, j+1) - 
                                 self.Jprfunc(i, j+1, j+1, True) * self.Nfunc(i, j, j+1))
                                 
    def calculate_impedances(self):
        """Расчет импедансов и адмитансов"""
        for i in range(self.lens.toch - 1):
            for h in range(len(self.lens.a)):
                if h == 0:
                    numerator = cmath.exp(self.alfa[1] * 1j) * abs(self.lens.eps[1])
                    denominator = cmath.exp(self.alfa[0] * 1j) * abs(self.lens.eps[0])
                    self.Z[i][h] = cmath.sqrt(numerator / denominator) * (self.Jpr[i] / self.J[i])
                    numerator = cmath.exp(self.alfa[0] * 1j) * abs(self.lens.eps[0])
                    denominator = cmath.exp(self.alfa[1] * 1j) * abs(self.lens.eps[1])
                    self.Y[i][h] = cmath.sqrt(numerator / denominator) * (self.Jpr[i] / self.J[i])
                    
                elif h == (len(self.lens.a) - 1):
                    numerator = cmath.exp(self.alfa[h+1] * 1j) * abs(self.lens.eps[h+1])
                    denominator = cmath.exp(self.alfa[h] * 1j) * abs(self.lens.eps[h])
                    sqrt_part = cmath.sqrt(numerator / denominator) 
                
                    term1 = self.Cpr[i][h-1] + self.Z[i][h-1] * self.Spr[i][h-1]
                    term2 = self.C[i][h-1] + self.Z[i][h-1] * self.S[i][h-1]
                    self.Z[i][h] = sqrt_part * (term1 / term2) 
                
                    numerator = cmath.exp(self.alfa[h] * 1j) * abs(self.lens.eps[h])
                    denominator = cmath.exp(self.alfa[h+1] * 1j) * abs(self.lens.eps[h+1])
                    sqrt_part = cmath.sqrt(numerator / denominator)
                    term1 = self.Cpr[i][h-1] + self.Y[i][h-1] * self.Spr[i][h-1]
                    term2 = self.C[i][h-1] + self.Y[i][h-1] * self.S[i][h-1]
                    self.Y[i][h] = sqrt_part * (term1 / term2) 
                
                else:
                    numerator = cmath.exp(self.alfa[h+1] * 1j) * abs(self.lens.eps[h+1])
                    denominator = cmath.exp(self.alfa[h] * 1j) * abs(self.lens.eps[h])
                    sqrt_part = cmath.sqrt(numerator / denominator)
                
                    term1 = self.Cpr[i][h-1] + self.Z[i][h-1] * self.Spr[i][h-1]
                    term2 = self.C[i][h-1] + self.Z[i][h-1] * self.S[i][h-1]
                    self.Z[i][h] = sqrt_part * (term1 / term2)

                    numerator = cmath.exp(self.alfa[h] * 1j) * abs(self.lens.eps[h])
                    denominator = cmath.exp(self.alfa[h+1] * 1j) * abs(self.lens.eps[h+1])
                    sqrt_part = cmath.sqrt(numerator / denominator)

                    term1 = self.Cpr[i][h-1] + self.Y[i][h-1] * self.Spr[i][h-1]
                    term2 = self.C[i][h-1] + self.Y[i][h-1] * self.S[i][h-1]
                    self.Y[i][h] = sqrt_part * (term1 / term2)
                    
    def Hfunc(self, i, k1):
        """Функция Ханкеля второго рода"""
        nu = i + 1
        H = (scipy.special.hankel1(nu + 0.5, k1)) * (cmath.sqrt(k1 * math.pi/2))
        return H
    
    def Hprfunc(self, i, k1):
        """Производная функции Ханкеля второго рода"""
        nu = i + 1
        Hpr = ((nu / (2 * nu + 1)) * (((scipy.special.hankel1(nu - 0.5, k1) * (cmath.sqrt(k1 * math.pi/2))) / k1)) * k1 - \
        (((nu + 1) / (2 * nu + 1)) * ((scipy.special.hankel1(nu + 1.5, k1)) * (cmath.sqrt(k1 * math.pi/2))) / k1) * k1 + \
        ((scipy.special.hankel1(nu + 0.5, k1)) * (cmath.sqrt(k1 * math.pi/2))) / k1)
        return Hpr

    
    def calculate_modified_functions(self):
        """Расчет модифицированных функций"""
        k1 = self.lens.k0
        k00 = self.k[0][0]
        self.k[0][0] = self.lens.k0
        
        for i in range(self.lens.toch):
            self.mJ[i] = self.Jfunc(i, 0, 0)
            self.mJpr[i] = self.Jprfunc(i, 0, 0, True)
            self.mH[i] = self.Hfunc(i, self.lens.k0)
            self.mHpr[i] = self.Hprfunc(i, self.lens.k0)
            
        self.lens.k0 = k1
        self.k[0][0] = k00
    
    # нужно сделать выбор между гюгенса и рупором
    def calculate_scattering_coefficients(self):
        """Расчет коэффициентов рассеяния"""
        for i in range(self.lens.toch):
            n = i + 1
            # Элемент Гюгенса дифракция
            # self.Mn[i] = (self.Z[i][self.lens.n-1] * self.mJ[i] - self.mJpr[i]) / (self.Z[i][self.lens.n-1] * self.mH[i] - self.mHpr[i])
            # self.Nn[i] = (self.Y[i][self.lens.n-1] * self.mJ[i] - self.mJpr[i]) / (self.Y[i][self.lens.n-1] * self.mH[i] - self.mHpr[i])
            
            # Рупор на поверхности сферы
            self.Mn[i] = (self.Z[i][self.lens.n-1] - 1j) / (self.Z[i][self.lens.n-1] * self.mH[i] - self.mHpr[i])             
            self.Nn[i] = (self.Y[i][self.lens.n-1] - 1j) / (self.Y[i][self.lens.n-1] * self.mH[i] - self.mHpr[i])
            self.Mn[i] = self.Mn[i].real - self.Mn[i].imag * 1j
            self.Nn[i] = self.Nn[i].real - self.Nn[i].imag * 1j
        
    def calculate_angles(self):
        """Расчет углов для диаграммы направленности"""
        for i in range(self.steps):
            if i == 0:
                self.teta[i] = self.teta_start * (math.pi/180)
            else:
                self.teta[i] = self.teta[i-1] + self.step
            self.cos_teta[i] = math.cos(self.teta[i])
    
    def calculate_legendre_functions(self):
        """Расчет функций Лежандра"""
        self.pii = np.zeros((self.lens.toch+1, 2*self.steps+1))
        self.tay = np.zeros((self.lens.toch+1, 2*self.steps+1))
        
        for i in range(self.lens.toch):
            m = i + 1
            M = scipy.special.lpmv(0, m, self.cos_teta)
            Lm0 = M
            M = scipy.special.lpmv(1, m, self.cos_teta)
            Lm1 = M
            
            if m < 2:
                Lm2 = 0
            else:
                M = scipy.special.lpmv(2, m, self.cos_teta)
                Lm2 = M
            
            for z in range(len(self.teta)):
                if (self.teta[z] > 0) and (self.teta[z] < math.pi):
                    self.pii[i][z] = (1 * Lm1[z]) / math.sin(self.teta[z])
                elif (self.teta[z] > math.pi) and (self.teta[z] < 2*math.pi):
                    self.pii[i][z] = (-1 * Lm1[z]) / math.sin(self.teta[z])
            
            for z in range(len(self.teta)):
                if m < 2:
                    self.tay[i][z] = 0.5 * (-m * (m + 1) * Lm0[z])
                else:
                    self.tay[i][z] = 0.5 * (Lm2[z] - m * (m + 1) * Lm0[z])
    
    def calculate_circular_polarization(self):
        """Расчет полей для круговой поляризации"""
        for z in range(len(self.teta)):
            for p in range(self.lens.toch):
                y = p + 1
                self.E_op[p][z] = ((2*y + 1) / (y*(y + 1))) * ((-1)**y) * (self.tay[p][z] - self.pii[p][z]) * (self.Mn[p] + self.Nn[p])
                self.E_kp[p][z] = ((2*y + 1) / (y*(y + 1))) * ((-1)**y) * (self.tay[p][z] + self.pii[p][z]) * (self.Mn[p] - self.Nn[p])
        
        self.P1 = np.sum(self.E_op, axis=0)
        self.P2 = np.sum(self.E_kp, axis=0)
        self.Pab1 = np.real(self.P1)
        self.Pab2 = np.real(self.P2)
    
    def calculate_linear_polarization(self):
        """Расчет полей для линейной поляризации"""
        for z in range(len(self.teta)):
            for p in range(self.lens.toch):
                y = p + 1
                self.S_teta[p][z] = ((2*y + 1)/(y*(y + 1))) * ((-1)**y) * (-1 * (self.tay[p][z] * self.Mn[p] - self.pii[p][z] * self.Nn[p]) * 
                                     math.cos(self.teta[z]) * math.cos(self.lens.phi)**2 - (self.pii[p][z] * self.Mn[p] - self.tay[p][z] * self.Nn[p]) * 
                                     math.sin(self.lens.phi)**2)
                self.S_phi[p][z] = ((2*y + 1)/(y*(y + 1))) * ((-1)**y) * ((self.tay[p][z] * self.Mn[p] - self.pii[p][z] * self.Nn[p]) * 
                                    math.cos(self.lens.phi) * math.sin(self.lens.phi) - (self.pii[p][z] * self.Mn[p] - self.tay[p][z] * self.Nn[p]) * 
                                    math.cos(self.teta[z])**2 * math.sin(self.lens.phi)*math.cos(self.lens.phi))
                self.E_teta[0][z] += self.S_teta[p][z]
                self.E_phi[0][z] += self.S_phi[p][z]
            
            self.E_teta[0][z] = (1 - (math.sin(self.teta[z]) * math.cos(self.lens.phi))**2)**(-0.5) * self.E_teta[0][z]
            self.E_phi[0][z] = (1 - (math.sin(self.teta[z]) * math.cos(self.lens.phi))**2)**(-0.5) * self.E_phi[0][z]

            for p in range(self.lens.toch):
                self.E_teta[0][z] = abs(self.E_teta[0][z])
                self.E_phi[0][z] = abs(self.E_phi[0][z])
    
    def normalize_results(self):
        """Нормализация результатов"""
        self.tetay = np.zeros(self.steps)
        
        for i in range(self.steps):
            self.teta[i] = self.teta[i] 
        
        for i in range(len(self.teta)):
            self.tetay[i] = (self.teta[i] * (self.steps / (2 * math.pi)))
        
        # Нормированные E к k0a
        self.DN_NORM_lin_k0a_teta = self.E_teta[0] / self.lens.k0
        self.DN_NORM_lin_k0a_phi = self.E_phi[0] / self.lens.k0
        self.DN_NORM_circle_k0a_op = self.Pab1 / self.lens.k0
        self.DN_NORM_circle_k0a_kp = self.Pab2 / self.lens.k0
        
        # Нормированные E dB
        E_teta_max = np.max(self.E_teta[0])
        E_phi_max = np.max(self.E_phi[0])
        Pab1_max = np.max(self.Pab1)
        Pab2_max = np.max(self.Pab2)
        
        self.DN_NORM_lin_dB_teta = 20 * np.log10(self.E_teta[0] / E_teta_max)
        self.DN_NORM_lin_dB_phi = 20 * np.log10(self.E_phi[0] / E_phi_max)
        self.DN_NORM_circle_dB_op = 20 * np.log10(self.Pab1 / Pab1_max)
        self.DN_NORM_circle_dB_kp = 20 * np.log10(self.Pab2 / Pab2_max)
    
    def run_calculation(self):
        """Выполнение полного расчета"""
        
        self.calculate_medium_parameters()
        self.calculate_k_coefficients()
        self.calculate_bessel_functions()
        self.calculate_CS_functions()
        self.calculate_impedances()
        self.calculate_modified_functions()
        self.calculate_scattering_coefficients()
        self.calculate_angles()
        self.calculate_legendre_functions()
        self.calculate_circular_polarization()
        self.calculate_linear_polarization()
        self.normalize_results()