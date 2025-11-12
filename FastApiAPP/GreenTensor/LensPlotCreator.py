import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib.ticker as ticker
from typing import Optional, Tuple, List, Union
from scipy.stats import pearsonr
from .LensCalculator import LensCalculator

class LensPlotCreator:
    """Класс для создания графиков для анализа линз"""
    
    def __init__(self, calculator: LensCalculator):
        """
        Инициализация класса для построения графиков.
        
        Parameters:
        -----------
        calculator : LensCalculator
            Калькулятор с рассчитанными данными
        """
        self.calc = calculator
        
        self.theta_original = calculator.teta
        self.E1_original = calculator.DN_NORM_lin_dB_teta
        self.E2 = None
        
        self.phi = calculator.lens.phi
        
        self.comp = "θ" if self.phi == 0 else "φ"
        
        self.comparison_mode = False
        self.ref_calc = None
        
        self.theta = self.theta_original.copy()
        self.E1 = self.E1_original.copy()
        
    def setup_comparison(self, reference: Union[LensCalculator, np.ndarray]) -> None:
        """
        Настройка режима сравнения с другим калькулятором или данными.
        
        Parameters:
        -----------
        reference : Union[LensCalculator, np.ndarray, List]
            Референсный калькулятор или массив данных для сравнения
        """
        self.comparison_mode = True
        
        self.theta = self.theta_original.copy()
        self.E1 = self.E1_original.copy()
        
        if isinstance(reference, LensCalculator):
            self.ref_calc = reference
            self.E2_shifted = np.real(reference.DN_NORM_lin_dB_teta)
        elif isinstance(reference, (np.ndarray, list)):
            self.ref_calc = None
            self.E2_shifted = np.real(reference)
        else:
            raise TypeError(f"reference должен быть LensCalculator, numpy.ndarray или list, получен {type(reference)}")
        
        self.E1_shifted = np.real(self.E1)  # Наши расчеты

        size_E1 = len(self.E1_shifted)
        size_E2 = len(self.E2_shifted)
        
        if size_E1 != size_E2:
            min_size = min(size_E1, size_E2)
            self.E1_shifted = self.E1_shifted[:min_size]
            self.E2_shifted = self.E2_shifted[:min_size]
            self.theta = self.theta[:min_size]
            self.E1 = self.E1[:min_size]
        
        E1_max = np.max(self.E1_shifted)
        self.valid_mask = self.E1_shifted >= (E1_max - 3)
        
        self.squared_errors = np.zeros_like(self.E1_shifted)
        self.squared_errors3 = np.zeros_like(self.E1_shifted)

        self.squared_errors[self.valid_mask] = np.sqrt((self.E1_shifted[self.valid_mask] - self.E2_shifted[self.valid_mask]) ** 2)
        self.squared_errors3 = np.sqrt(np.abs((self.E1_shifted - self.E2_shifted) ** 2))

        self.mse = np.mean(self.squared_errors[self.valid_mask]) if np.any(self.valid_mask) else 0
        self.mseMax = np.mean(self.squared_errors3)

        self.Ymin = -70
        
        print("Сравнение успешно настроено")
    
    def plot_single_polar(self) -> plt.Figure:
        """Полярный график для одного калькулятора (Eθ и Eφ)."""
        theta = self.theta_original
        E1 = self.E1_original
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        ax.set_ylim(-40, 0)
        
        ax.plot(theta, np.real(E1), 
                color='#1f77b4', linestyle='--', linewidth=1.5, 
                alpha=0.9, label=r'E$_\theta$($\theta$), дБ')
        
        if (self.E2 is not None):
            ax.plot(theta, np.real(self.E2), 
                    color='#ff7f0e', linestyle='-', linewidth=2.5, 
                    alpha=0.9, label=r'E$_\varphi$($\theta$), дБ')
        
        ax.legend(loc='best', fontsize=14, frameon=True, framealpha=0.95)
        ax.set_theta_zero_location('E')  # 0° сверху
        ax.set_theta_direction(1)       # По часовой стрелке
        
        ax.text(0, ax.get_rmax() + 7, r'$\theta$$\degree$', fontsize=14, ha='right', va='center')
        plt.tight_layout()
        
        return fig
    
    def plot_comparison_polar(self) -> plt.Figure:
        """Полярный график сравнения двух калькуляторов."""
        if not self.comparison_mode:
            raise ValueError("Режим сравнения не активирован. Используйте setup_comparison()")
            
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
        ax.set_ylim(-40, 0)
        
        ax.plot(self.theta, np.real(self.E1), 
                color='#1f77b4', linestyle='--', linewidth=1.5, 
                alpha=0.9, label=r'E$_\theta$($\theta$), дБ')
        
        ax.plot(self.theta, self.E2_shifted, 
                color='#ff7f0e', linestyle='-', linewidth=2.5, 
                alpha=0.9, label=r'Reference data, дБ')
        
        ax.legend(loc='best', fontsize=14, frameon=True, framealpha=0.95)
        ax.set_theta_zero_location('E')  # 0° сверху
        ax.set_theta_direction(1)       # По часовой стрелке
        
        ax.text(0, ax.get_rmax() + 7, r'$\theta$$\degree$', fontsize=14, ha='right', va='center')
        plt.tight_layout()
        
        return fig
    
    def plot_single(self) -> List[plt.Figure]:
        """Построение графиков для одного калькулятора."""
        figs = []
        figs.append(self.plot_single_polar())
        plt.show()
        return figs
    
    def plot_comparison(self) -> List[plt.Figure]:
        """Построение всех графиков сравнения."""
        if not self.comparison_mode:
            raise ValueError("Режим сравнения не активирован. Используйте setup_comparison()")
            
        figs = []
        figs.append(self.plot_comparison_polar())
        
        plt.show()
        return figs