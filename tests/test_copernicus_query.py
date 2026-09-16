import pytest
from terraresolve.ingest.copernicus import search_products

def test_copernicus_query_returns_results():
    """
    Ensure the OData catalog filter properly retrieves Sentinel-2 L2A 
    products. Previously, the productType filter was malformed, causing
    all queries to return 0 results.
    """
    # A known bbox and date range in Bangalore that contains products
    products = search_products(
        bbox=(77.55, 13.07, 77.62, 13.15),
        start_date="2026-01-01",
        end_date="2026-09-01",
        max_cloud_cover=20.0
    )
    
    assert len(products) > 0, "Expected at least one product to match"
    
    first_product = products[0]
    assert "MSIL2A" in first_product["Name"], "Expected L2A product"
