from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from GreenTensor.Lens import Lens
from GreenTensor.LensCalculator import LensCalculator
from GreenTensor.LensPlotCreator import LensPlotCreator
import matplotlib.pyplot as plt
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import zipfile
import numpy as np
import math

app = FastAPI(title="Green Tensor Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

class LensConfiguration(BaseModel):
    """Конфигурация линзы для расчета"""
    wave_number: float = Field(default=4 * math.pi, gt=0, description="Волновое число k0")
    series_terms: int = Field(default=20, ge=1, le=200, description="Количество членов ряда (toch)")
    layers_count: int = Field(..., ge=2, le=10, description="Число слоев линзы")
    azimuth_angle: float = Field(default=1.5707963267948966, ge=0, le=6.283185307179586, description="Азимутальный угол φ в радианах")
    radii: List[float] = Field(..., description="Радиусы слоев (нормированные)")
    dielectric_constants: List[complex] = Field(..., description="Комплексные диэлектрические проницаемости")
    magnetic_permeabilities: List[complex] = Field(..., description="Комплексные магнитные проницаемости")
    alternative_wave_number: Optional[float] = Field(None, description="Альтернативное волновое число k1")

class PlotRequest(BaseModel):
    """Запрос на генерацию графиков"""
    lens_config: LensConfiguration
    plot_type: Literal["linear", "circular", "both"] = Field("both", description="Тип поляризации")
    output_format: Literal["polar", "cartesian", "both"] = Field("both", description="Тип графика")
    normalize: bool = Field(True, description="Нормализовать результаты")
    angle_range: tuple[float, float] = Field((0.01, 6.283185307179586), description="Диапазон углов в радианах")

class CalculationResult(BaseModel):
    """Результаты расчета"""
    angles_rad: List[float]
    angles_deg: List[float]
    linear_theta: List[float]
    linear_phi: List[float]
    circular_op: List[float]
    circular_kp: List[float]
    linear_dB_theta: List[float]
    linear_dB_phi: List[float]
    circular_dB_op: List[float]
    circular_dB_kp: List[float]

def create_lens_from_config(config: LensConfiguration) -> Lens:
    """Создание объекта линзы из конфигурации"""
    return Lens(
        k0=config.wave_number,
        toch=config.series_terms,
        n=config.layers_count,
        phi=config.azimuth_angle,
        a=config.radii,
        eps=config.dielectric_constants,
        miy=config.magnetic_permeabilities,
        k1=config.alternative_wave_number
    )

@app.post("/calculate", response_model=CalculationResult)
async def calculate_lens_properties(request: PlotRequest):
    """
    Выполнить расчет характеристик линзы
    """
    try:
        # Создание и расчет линзы
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()
        
        # Преобразование углов в градусы
        angles_deg = [angle * 180 / np.pi for angle in calculator.teta]
        
        return CalculationResult(
            angles_rad=calculator.teta.tolist(),
            angles_deg=angles_deg,
            linear_theta=calculator.E_teta[0].tolist(),
            linear_phi=calculator.E_phi[0].tolist(),
            circular_op=calculator.Pab1.tolist(),
            circular_kp=calculator.Pab2.tolist(),
            linear_dB_theta=calculator.DN_NORM_lin_dB_teta.tolist(),
            linear_dB_phi=calculator.DN_NORM_lin_dB_phi.tolist(),
            circular_dB_op=calculator.DN_NORM_circle_dB_op.tolist(),
            circular_dB_kp=calculator.DN_NORM_circle_dB_kp.tolist()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "config": request.lens_config.dict()
            }
        )

@app.post("/generate-plots/")
async def generate_plots(request: PlotRequest):
    """
    Сгенерировать графики для линзы
    """
    try:
        # Создание и расчет линзы
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()
        
        # Создание графиков
        plot_creator = LensPlotCreator(calculator)
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            # Генерация графиков в зависимости от запроса
            if request.output_format in ["polar", "both"]:
                fig_polar = plot_creator.plot_single_polar()
                polar_buffer = BytesIO()
                fig_polar.savefig(polar_buffer, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig_polar)
                polar_buffer.seek(0)
                zip_file.writestr("polar_diagram.png", polar_buffer.getvalue())
            
            # Для cartesian графиков можем создать дополнительную визуализацию
            if request.output_format in ["cartesian", "both"]:
                fig_cartesian = _create_cartesian_plot(calculator, request.plot_type)
                cartesian_buffer = BytesIO()
                fig_cartesian.savefig(cartesian_buffer, format='png', bbox_inches='tight', dpi=150)
                plt.close(fig_cartesian)
                cartesian_buffer.seek(0)
                zip_file.writestr("cartesian_diagram.png", cartesian_buffer.getvalue())
            
            # Добавляем CSV с данными
            csv_data = _create_csv_data(calculator)
            zip_file.writestr("calculation_data.csv", csv_data)
            
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=lens_analysis.zip",
                "Content-Length": str(zip_buffer.getbuffer().nbytes)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "config": request.lens_config.dict()
            }
        )

@app.post("/compare-lenses/")
async def compare_lenses(config1: LensConfiguration, config2: LensConfiguration):
    """
    Сравнить две конфигурации линз
    """
    try:
        # Расчет первой линзы
        lens1 = create_lens_from_config(config1)
        calc1 = LensCalculator(lens1)
        calc1.run_calculation()
        
        # Расчет второй линзы
        lens2 = create_lens_from_config(config2)
        calc2 = LensCalculator(lens2)
        calc2.run_calculation()
        
        # Создание графиков сравнения
        plot_creator = LensPlotCreator(calc1)
        plot_creator.setup_comparison(calc2)
        
        fig_comparison = plot_creator.plot_comparison_polar()
        
        comparison_buffer = BytesIO()
        fig_comparison.savefig(comparison_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig_comparison)
        comparison_buffer.seek(0)
        
        return StreamingResponse(
            comparison_buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=lens_comparison.png"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "type": type(e).__name__
            }
        )

def _create_cartesian_plot(calculator: LensCalculator, plot_type: str) -> plt.Figure:
    """Создание декартового графика"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    angles_deg = [angle * 180 / np.pi for angle in calculator.teta]
    
    if plot_type in ["linear", "both"]:
        ax.plot(angles_deg, calculator.DN_NORM_lin_dB_teta, 
                label='Eθ (dB)', linewidth=2, alpha=0.8)
        ax.plot(angles_deg, calculator.DN_NORM_lin_dB_phi, 
                label='Eφ (dB)', linewidth=2, alpha=0.8)
    
    if plot_type in ["circular", "both"]:
        ax.plot(angles_deg, calculator.DN_NORM_circle_dB_op, 
                label='Circular OP (dB)', linewidth=2, alpha=0.8, linestyle='--')
        ax.plot(angles_deg, calculator.DN_NORM_circle_dB_kp, 
                label='Circular KP (dB)', linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Normalized Field (dB)')
    ax.set_title('Lens Radiation Pattern')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def _create_csv_data(calculator: LensCalculator) -> str:
    """Создание CSV с данными расчета"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Заголовок
    writer.writerow([
        'angle_rad', 'angle_deg', 
        'E_theta', 'E_phi', 'E_circular_op', 'E_circular_kp',
        'E_theta_dB', 'E_phi_dB', 'E_circular_op_dB', 'E_circular_kp_dB'
    ])
    
    # Данные
    angles_deg = [angle * 180 / np.pi for angle in calculator.teta]
    for i in range(len(calculator.teta)):
        writer.writerow([
            f"{calculator.teta[i]:.6f}",
            f"{angles_deg[i]:.2f}",
            f"{calculator.E_teta[0][i]:.6f}",
            f"{calculator.E_phi[0][i]:.6f}",
            f"{calculator.Pab1[i]:.6f}",
            f"{calculator.Pab2[i]:.6f}",
            f"{calculator.DN_NORM_lin_dB_teta[i]:.6f}",
            f"{calculator.DN_NORM_lin_dB_phi[i]:.6f}",
            f"{calculator.DN_NORM_circle_dB_op[i]:.6f}",
            f"{calculator.DN_NORM_circle_dB_kp[i]:.6f}"
        ])
    
    return output.getvalue()

@app.get("/")
async def root():
    return {
        "message": "GreenTensor Lens Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "calculate": "POST /calculate - Расчет характеристик линзы",
            "generate_plots": "POST /generate-plots - Генерация графиков",
            "compare_lenses": "POST /compare-lenses - Сравнение линз",
            "docs": "GET /docs - Документация API"
        }
    }