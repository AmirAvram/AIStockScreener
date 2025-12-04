from pydantic import BaseModel


class Stock(BaseModel):
    name: str
    ticker: str
    summary: str
    current_price: float
    upside_percent: float


class StocksPick(BaseModel):
    stocks: list[Stock]


class StocksReport(BaseModel):
    date: str
    stocks: StocksPick
