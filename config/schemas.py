"""Pydantic models for data validation."""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Literal


class SalesRecord(BaseModel):
    """Schema for a single sales record."""

    date: datetime
    region: str = Field(min_length=2)
    model: str = Field(min_length=1)
    sales: int = Field(ge=0)
    price: float = Field(gt=0)

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        """Validate region code.

        Args:
            v: Region string to validate.

        Returns:
            Validated uppercase region code.

        Raises:
            ValueError: If region is not in valid list.
        """
        valid_regions = ["NA", "EU", "ASIA", "LATAM", "MEA"]
        if v.upper() not in valid_regions:
            raise ValueError(f"Region must be one of {valid_regions}")
        return v.upper()


class DataProfile(BaseModel):
    """Schema for data profile output."""

    total_records: int
    date_range: tuple[datetime, datetime]
    total_sales: int
    avg_price: float
    regions: list[str]
    models: list[str]


class PlotSpecification(BaseModel):
    """Schema for plot specification."""

    plot_type: Literal["line", "bar", "scatter", "heatmap", "box"]
    title: str
    x_axis: str
    y_axis: str
    groupby: Optional[str] = None
    color: Optional[str] = None
    description: str

