from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from GreenTensor.Lens import Lens
from GreenTensor.LensCalculator import LensCalculator
from GreenTensor.LensPlotCreator import LensPlotCreator
import matplotlib.pyplot as plt
from io import BytesIO
from pydantic import BaseModel, Field
from typing import List
import zipfile

app = FastAPI(title="Green Tensor Image Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)

class LensParameters(BaseModel):
    """
    Parameters for lens generation and plotting.
    
    Attributes:
        radiusRatio: Радиус линзы (коэффициент умножения pi)
        layers_count: Число слоев линзы (последний слой - воздух)
        norm_radii: Нормированные радиусы слоев
        dielectric_constants: Диэлектрическая проницаемость материала слоев
        magnetic_permeabilities: Магнитная проницаемость материала слоев
        plot_type: Типы изображений для генерации - "line", "polar" или "both"
    """
    radiusRatio: int = Field(..., gt=0, description="Радиус линзы (коэффициент умножения pi)")
    layers_count: int = Field(..., gt=0, description="Число слоев линзы (последний слой - воздух)")
    norm_radii: List[float] = Field(..., description="Нормированные радиусы слоев")
    dielectric_constants: List[float] = Field(..., description="Диэлектрическая проницаемость материала слоев")
    magnetic_permeabilities: List[float] = Field(..., description="Магнитная проницаемость материала слоев")
    plot_type: str = Field("both", description="Типы изображений для генерации - 'line', 'polar' или 'both'", 
                          pattern="^(line|polar|both)$")

@app.post("/generate-images/")
async def generate_images(params: LensParameters):
    try:
        lens = Lens(params.radiusRatio, params.layers_count, params.norm_radii, params.dielectric_constants, params.magnetic_permeabilities)
        lensCalc = LensCalculator(lens)
        fig_line, fig_polar = LensPlotCreator.create_plots(lensCalc)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if params.plot_type in ["line", "both"]:
                line_buffer = BytesIO()
                fig_line.savefig(line_buffer, format='png', bbox_inches='tight')
                plt.close(fig_line)
                line_buffer.seek(0)
                zip_file.writestr("lens_line.png", line_buffer.getvalue())

            if params.plot_type in ["polar", "both"]:
                polar_buffer = BytesIO()
                fig_polar.savefig(polar_buffer, format='png', bbox_inches='tight')
                plt.close(fig_polar)
                polar_buffer.seek(0)
                zip_file.writestr("lens_polar.png", polar_buffer.getvalue())
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=images.zip",
                "Content-Length": str(zip_buffer.getbuffer().nbytes)
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "message": getattr(e, "message", str(e))
            }
        )

@app.get("/")
async def read_root():
    return {
        "message": "Добро пожаловать в генератор изображений линз",
        "instructions": "Документация: /docs",
    }