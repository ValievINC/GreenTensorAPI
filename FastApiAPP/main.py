from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from GreenTensor.Lens import Lens
from GreenTensor.LensCalculator import LensCalculator
from GreenTensor.LensPlotCreator import LensPlotCreator
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
from pydantic import BaseModel, Field
from typing import List, Optional
import zipfile
import numpy as np
import pandas as pd
import csv
from fastapi import UploadFile, File, Form
import json
from io import StringIO


app = FastAPI(title="Green Tensor Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LensConfiguration(BaseModel):
    wave_number: float = Field(..., description="Волновое число")
    series_terms: int = Field(..., description="Количество членов ряда")
    layers_count: int = Field(..., description="Количество слоев")
    azimuth_angle: float = Field(..., description="Азимутальный угол")
    radii: List[float] = Field(..., description="Радиусы слоев")
    dielectric_constants: List[float] = Field(..., description="Диэлектрические постоянные")
    magnetic_permeabilities: List[float] = Field(..., description="Магнитные проницаемости")
    alternative_wave_number: Optional[float] = Field(None, description="Альтернативное волновое число")


class PlotRequest(BaseModel):
    lens_config: LensConfiguration
    normalize: bool = Field(True, description="Нормализовать результаты")


class CalculationResult(BaseModel):
    angles_rad: List[float]
    angles_deg: List[float]
    linear_theta: List[float]
    linear_phi: List[float]
    linear_dB_theta: List[float]
    linear_dB_phi: List[float]


def create_lens_from_config(config: LensConfiguration) -> Lens:
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
        raise HTTPException(status_code=400, detail=f"Ошибка расчета: {str(e)}")


@app.post("/generate-plot/")
async def generate_plot(request: PlotRequest):
    try:
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()

        plot_creator = LensPlotCreator(calculator)
        fig = plot_creator.plot_single_polar()

        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=polar_diagram.png"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка генерации графика: {str(e)}")


@app.post("/generate-analysis/")
async def generate_analysis(request: PlotRequest):
    try:
        lens = create_lens_from_config(request.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as archive:

            plot_creator = LensPlotCreator(calculator)
            fig = plot_creator.plot_single_polar()

            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150)
            plt.close(fig)
            img_buffer.seek(0)

            archive.writestr("polar_diagram.png", img_buffer.getvalue())
            archive.writestr("calculation_data.csv", _create_csv_data(calculator))

        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=lens_analysis.zip"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка генерации анализа: {str(e)}")


def _create_csv_data(calculator: LensCalculator) -> str:
    output = StringIO()
    writer = csv.writer(output)

    writer.writerow(['angle_rad', 'angle_deg', 'E_theta', 'E_phi', 'E_theta_dB', 'E_phi_dB'])
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


@app.post("/compare-csv/")
async def compare_csv(
    file: UploadFile = File(...),
    request: str = Form(...)
):
    try:
        # Парсим request из FormData
        request_data = json.loads(request)
        request_model = PlotRequest(**request_data)

        # Читаем CSV
        csv_content = await file.read()
        df = pd.read_csv(StringIO(csv_content.decode('utf-8')))

        # Берём последнюю колонку (значения dB)
        reference_array = df.iloc[1:, -1].to_numpy().astype(float)

        # Выполняем расчёт
        lens = create_lens_from_config(request_model.lens_config)
        calculator = LensCalculator(lens)
        calculator.run_calculation()

        # Сравнение
        plotter = LensPlotCreator(calculator)
        plotter.setup_comparison(reference_array)

        fig = plotter.plot_comparison_polar()

        out = BytesIO()
        fig.savefig(out, format="png", dpi=150)
        plt.close(fig)
        out.seek(0)

        return StreamingResponse(
            out,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=comparison.png"}
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
async def root():
    return {
        "message": "GreenTensor Lens Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "calculate": "POST /calculate",
            "generate_plot": "POST /generate-plot",
            "generate_analysis": "POST /generate-analysis",
            "compare_csv": "POST /compare-csv",
            "docs": "GET /docs"
        }
    }
