import math
from typing import List, Optional, Union, Tuple


class Lens:
    """
    Класс для представления линзы

    Параметры:
    ----------
    k0 : float, optional
        Волновое число (по умолчанию 4π)
    toch : int, optional
        Количество учитываемых членов ряда (по умолчанию 20)
    n : int, optional
        Число слоев (последний слой - воздух, по умолчанию 4)
    phi : float, optional
        Азимутальный угол в радианах (по умолчанию π/2)
    a : List[float], optional
        Радиусы слоев сферы
    eps : List[complex], optional
        Комплексные диэлектрические проницаемости слоев
    miy : List[complex], optional
        Комплексные магнитные проницаемости слоев
    k1 : float, optional
        Альтернативное волновое число для некоторых расчетов
    """
    def __init__(
        self,
        k0: float = 4 * math.pi,
        toch: int = 20,
        n: int = 4,
        phi: float = math.pi / 2,
        a: Optional[List[float]] = None,
        eps: Optional[List[complex]] = None,
        miy: Optional[List[complex]] = None,
        k1: Optional[float] = None
    ):
        # Основные параметры
        self.k0 = k0
        self.k1 = k1 if k1 is not None else k0
        self.toch = toch
        self.n = n
        self.phi = phi
        
        # Параметры материалов
        self.a = a if a is not None else [0.53, 0.75, 0.93, 1.0]
        self.eps = eps if eps is not None else [1.86, 1.57, 1.28, 1.0]
        self.miy = miy if miy is not None else [1.0, 1.0, 1.0, 1.0]
        
        # Валидация входных параметров
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Валидация всех входных параметров"""
        
        # Проверка типов данных
        if len(self.a) != self.n:
            raise ValueError(f"Длина списка радиусов a ({len(self.a)}) должна равняться n ({self.n})")
        if len(self.eps) != self.n:
            raise ValueError(f"Длина списка eps ({len(self.eps)}) должна равняться n ({self.n})")
        if len(self.miy) != self.n:
            raise ValueError(f"Длина списка miy ({len(self.miy)}) должна равняться n ({self.n})")
        
        if self.toch <= 0:
            raise ValueError("Количество членов ряда должно быть положительным")
        if self.k0 <= 0:
            raise ValueError("Волновое число должно быть положительным")