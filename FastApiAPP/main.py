from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from GreenTensor.Lens import Lens
from GreenTensor.LensCalculator import LensCalculator
from GreenTensor.LensPlotCreator import LensPlotCreator
import matplotlib.pyplot as plt
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List, Optional
import zipfile
import numpy as np

app = FastAPI(title="Green Tensor Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

class LensConfiguration(BaseModel):
    """Конфигурация линзы для расчета"""
    wave_number: float = Field(..., description="Волновое число")
    series_terms: int = Field(..., description="Количество членов ряда")
    layers_count: int = Field(..., description="Количество слоев")
    azimuth_angle: float = Field(..., description="Азимутальный угол")
    radii: List[float] = Field(..., description="Радиусы слоев")
    dielectric_constants: List[float] = Field(..., description="Диэлектрические постоянные (только действительные)")
    magnetic_permeabilities: List[float] = Field(..., description="Магнитные проницаемости (только действительные)")
    alternative_wave_number: Optional[float] = Field(None, description="Альтернативное волновое число")

class PlotRequest(BaseModel):
    """Запрос на генерацию графиков"""
    lens_config: LensConfiguration
    normalize: bool = Field(True, description="Нормализовать результаты")

class CalculationResult(BaseModel):
    """Результаты расчета"""
    angles_rad: List[float]
    angles_deg: List[float]
    linear_theta: List[float]
    linear_phi: List[float]
    linear_dB_theta: List[float]
    linear_dB_phi: List[float]

def create_lens_from_config(config: LensConfiguration) -> Lens:
    """Создание объекта линзы из конфигурации"""
    # Преобразуем в комплексные числа с нулевой мнимой частью
    eps_complex = [complex(val, 0.0) for val in config.dielectric_constants]
    miy_complex = [complex(val, 0.0) for val in config.magnetic_permeabilities]
    
    return Lens(
        k0=config.wave_number,
        toch=config.series_terms,
        n=config.layers_count,
        phi=config.azimuth_angle,
        a=config.radii,
        eps=eps_complex,
        miy=miy_complex,
        k1=config.alternative_wave_number
    )

@app.post("/calculate", response_model=CalculationResult)
async def calculate_lens_properties(request: PlotRequest):
    """
    Выполнить расчет характеристик линзы
    """
    try:
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()
        
        angles_deg = [angle * 180 / np.pi for angle in calculator.teta]

        return CalculationResult(
            angles_rad=calculator.teta.tolist(),
            angles_deg=angles_deg,
            linear_theta=calculator.E_teta[0].tolist(),
            linear_phi=calculator.E_phi[0].tolist(),
            linear_dB_theta=calculator.DN_NORM_lin_dB_teta.tolist(),
            linear_dB_phi=calculator.DN_NORM_lin_dB_phi.tolist()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка расчета: {str(e)}"
        )

@app.post("/generate-plot/")
async def generate_plot(request: PlotRequest):
    """
    Сгенерировать polar график для линзы используя LensPlotCreator
    """
    try:
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()
        
        plot_creator = LensPlotCreator(calculator)
        fig = plot_creator.plot_single_polar()
        
        plot_buffer = BytesIO()
        fig.savefig(plot_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        plot_buffer.seek(0)

        return StreamingResponse(
            plot_buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=polar_diagram.png"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка генерации графика: {str(e)}"
        )

@app.post("/generate-analysis/")
async def generate_analysis(request: PlotRequest):
    """
    Сгенерировать полный анализ с графиком и данными в ZIP
    """
    try:
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            
            plot_creator = LensPlotCreator(calculator)
            fig = plot_creator.plot_single_polar()
            plot_buffer = BytesIO()
            fig.savefig(plot_buffer, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)
            plot_buffer.seek(0)
            zip_file.writestr("polar_diagram.png", plot_buffer.getvalue())
            
            csv_data = _create_csv_data(calculator)
            zip_file.writestr("calculation_data.csv", csv_data)
            
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=lens_analysis.zip"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Ошибка генерации анализа: {str(e)}"
        )

def _create_csv_data(calculator: LensCalculator) -> str:
    """Создание CSV с данными расчета"""
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Заголовок
    writer.writerow([
        'angle_rad', 'angle_deg', 
        'E_theta', 'E_phi',
        'E_theta_dB', 'E_phi_dB'
    ])
    
    # Данные
    angles_deg = [angle * 180 / np.pi for angle in calculator.teta]
    for i in range(len(calculator.teta)):
        writer.writerow([
            f"{calculator.teta[i]:.6f}",
            f"{angles_deg[i]:.2f}",
            f"{calculator.E_teta[0][i]:.6f}",
            f"{calculator.E_phi[0][i]:.6f}",
            f"{calculator.DN_NORM_lin_dB_teta[i]:.6f}",
            f"{calculator.DN_NORM_lin_dB_phi[i]:.6f}"
        ])
    
    return output.getvalue()

@app.get("/")
async def root():
    return {
        "message": "GreenTensor Lens Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "calculate": "POST /calculate - Расчет характеристик линзы",
            "generate_plot": "POST /generate-plot - Генерация polar графика",
            "generate_analysis": "POST /generate-analysis - Полный анализ (график + данные)",
            "docs": "GET /docs - Документация API"
        }
    }