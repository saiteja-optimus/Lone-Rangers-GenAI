"""
Latitude/longitude lookup for major US cities that appear in FreightBrain load data.
Used as a fallback when coordinates are not embedded in the source record.
"""

from typing import Optional

# (lat, lon) tuples — WGS-84
CITY_COORDS: dict[str, tuple[float, float]] = {
    # Alabama
    "Birmingham,AL": (33.5186, -86.8104),
    "Mobile,AL": (30.6954, -88.0399),
    # Arizona
    "Phoenix,AZ": (33.4484, -112.0740),
    "Tucson,AZ": (32.2226, -110.9747),
    # California
    "Los Angeles,CA": (34.0522, -118.2437),
    "Oakland,CA": (37.8044, -122.2712),
    "Sacramento,CA": (38.5816, -121.4944),
    "San Diego,CA": (32.7157, -117.1611),
    "San Francisco,CA": (37.7749, -122.4194),
    "San Jose,CA": (37.3382, -121.8863),
    # Colorado
    "Denver,CO": (39.7392, -104.9903),
    "Colorado Springs,CO": (38.8339, -104.8214),
    # Florida
    "Fort Lauderdale,FL": (26.1224, -80.1373),
    "Jacksonville,FL": (30.3322, -81.6557),
    "Miami,FL": (25.7617, -80.1918),
    "Orlando,FL": (28.5383, -81.3792),
    "Tampa,FL": (27.9506, -82.4572),
    # Georgia
    "Atlanta,GA": (33.7490, -84.3880),
    "Savannah,GA": (32.0835, -81.0998),
    # Illinois
    "Chicago,IL": (41.8781, -87.6298),
    "Springfield,IL": (39.7817, -89.6501),
    # Indiana
    "Indianapolis,IN": (39.7684, -86.1581),
    # Iowa
    "Des Moines,IA": (41.5868, -93.6250),
    # Kansas
    "Kansas City,KS": (39.1155, -94.6268),
    "Wichita,KS": (37.6872, -97.3301),
    # Kentucky
    "Louisville,KY": (38.2527, -85.7585),
    "Lexington,KY": (38.0406, -84.5037),
    # Louisiana
    "New Orleans,LA": (29.9511, -90.0715),
    "Baton Rouge,LA": (30.4515, -91.1871),
    # Maryland
    "Baltimore,MD": (39.2904, -76.6122),
    # Massachusetts
    "Boston,MA": (42.3601, -71.0589),
    # Michigan
    "Detroit,MI": (42.3314, -83.0458),
    "Grand Rapids,MI": (42.9634, -85.6681),
    # Minnesota
    "Minneapolis,MN": (44.9778, -93.2650),
    "Saint Paul,MN": (44.9537, -93.0900),
    # Mississippi
    "Jackson,MS": (32.2988, -90.1848),
    # Missouri
    "Kansas City,MO": (39.0997, -94.5786),
    "St Louis,MO": (38.6270, -90.1994),
    "Saint Louis,MO": (38.6270, -90.1994),
    # Nebraska
    "Omaha,NE": (41.2565, -95.9345),
    # Nevada
    "Las Vegas,NV": (36.1699, -115.1398),
    "Reno,NV": (39.5296, -119.8138),
    # New Jersey
    "Edison,NJ": (40.5187, -74.4121),
    "Newark,NJ": (40.7357, -74.1724),
    # New Mexico
    "Albuquerque,NM": (35.0844, -106.6504),
    "Santa Fe,NM": (35.6870, -105.9378),
    # New York
    "Albany,NY": (42.6526, -73.7562),
    "Buffalo,NY": (42.8864, -78.8784),
    "New York,NY": (40.7128, -74.0060),
    # North Carolina
    "Charlotte,NC": (35.2271, -80.8431),
    "Raleigh,NC": (35.7796, -78.6382),
    # Ohio
    "Cincinnati,OH": (39.1031, -84.5120),
    "Cleveland,OH": (41.4993, -81.6944),
    "Columbus,OH": (39.9612, -82.9988),
    "Toledo,OH": (41.6639, -83.5552),
    # Oklahoma
    "Oklahoma City,OK": (35.4676, -97.5164),
    "Tulsa,OK": (36.1540, -95.9928),
    # Oregon
    "Portland,OR": (45.5051, -122.6750),
    "Eugene,OR": (44.0521, -123.0868),
    # Pennsylvania
    "Harrisburg,PA": (40.2732, -76.8867),
    "Philadelphia,PA": (39.9526, -75.1652),
    "Pittsburgh,PA": (40.4406, -79.9959),
    # Tennessee
    "Knoxville,TN": (35.9606, -83.9207),
    "Memphis,TN": (35.1495, -90.0490),
    "Nashville,TN": (36.1627, -86.7816),
    # Texas
    "Austin,TX": (30.2672, -97.7431),
    "Dallas,TX": (32.7767, -96.7970),
    "El Paso,TX": (31.7619, -106.4850),
    "Fort Worth,TX": (32.7555, -97.3308),
    "Houston,TX": (29.7604, -95.3698),
    "Lubbock,TX": (33.5779, -101.8552),
    "San Antonio,TX": (29.4241, -98.4936),
    # Utah
    "Salt Lake City,UT": (40.7608, -111.8910),
    "Provo,UT": (40.2338, -111.6585),
    # Virginia
    "Richmond,VA": (37.5407, -77.4360),
    "Virginia Beach,VA": (36.8529, -75.9780),
    # Washington
    "Seattle,WA": (47.6062, -122.3321),
    "Spokane,WA": (47.6588, -117.4260),
    "Tacoma,WA": (47.2529, -122.4443),
    # Wisconsin
    "Madison,WI": (43.0731, -89.4012),
    "Milwaukee,WI": (43.0389, -87.9065),
    # Washington DC
    "Washington,DC": (38.9072, -77.0369),

    # Additional cities found in load data
    # Arkansas
    "Little Rock,AR": (34.7465, -92.2896),
    # California (additional)
    "Bakersfield,CA": (35.3733, -119.0187),
    "Fontana,CA": (34.0922, -117.4350),
    "Fresno,CA": (36.7378, -119.7871),
    "Long Beach,CA": (33.7701, -118.1937),
    "Ontario,CA": (34.0633, -117.6509),
    "Stockton,CA": (37.9577, -121.2908),
    # Connecticut
    "Hartford,CT": (41.7637, -72.6851),
    "Bridgeport,CT": (41.1865, -73.1952),
    # Florida (additional)
    "Fort Lauderdale,FL": (26.1224, -80.1373),
    "Lakeland,FL": (28.0395, -81.9498),
    "Pensacola,FL": (30.4213, -87.2169),
    # Georgia (additional)
    "Augusta,GA": (33.4735, -82.0105),
    "Columbus,GA": (32.4610, -84.9877),
    # Illinois (additional)
    "Joliet,IL": (41.5250, -88.0817),
    "Rockford,IL": (42.2711, -89.0940),
    "Peoria,IL": (40.6936, -89.5890),
    # Indiana (additional)
    "Fort Wayne,IN": (41.1306, -85.1289),
    "Evansville,IN": (37.9716, -87.5711),
    # Maryland (additional)
    "Frederick,MD": (39.4143, -77.4105),
    # Missouri (additional — with period variant)
    "St. Louis,MO": (38.6270, -90.1994),
    # North Carolina (additional)
    "Greensboro,NC": (36.0726, -79.7920),
    "Winston-Salem,NC": (36.0999, -80.2442),
    "Durham,NC": (35.9940, -78.8986),
    # Pennsylvania (additional)
    "Allentown,PA": (40.6084, -75.4902),
    "Scranton,PA": (41.4090, -75.6624),
    "Erie,PA": (42.1292, -80.0851),
    "Reading,PA": (40.3356, -75.9269),
    # South Carolina
    "Charleston,SC": (32.7765, -79.9311),
    "Columbia,SC": (34.0007, -81.0348),
    "Greenville,SC": (34.8526, -82.3940),
    # Texas (additional)
    "Laredo,TX": (27.5064, -99.5075),
    "Amarillo,TX": (35.2220, -101.8313),
    "Corpus Christi,TX": (27.8006, -97.3964),
    "Beaumont,TX": (30.0860, -94.1018),
    # Virginia (additional)
    "Norfolk,VA": (36.8508, -76.2859),
    "Chesapeake,VA": (36.7682, -76.2875),
    "Hampton,VA": (37.0299, -76.3452),
}


def lookup_coords(city: str, state: str) -> Optional[tuple[float, float]]:
    """
    Return (lat, lon) for a city/state pair, or None if not in the lookup table.

    The key format is 'City,ST' — e.g. 'Atlanta,GA'.
    Matching is case-insensitive and strips extra whitespace.
    """
    key = f"{city.strip()},{state.strip().upper()}"
    result = CITY_COORDS.get(key)
    if result is not None:
        return result
    # Try title-cased city name as a fallback
    key_title = f"{city.strip().title()},{state.strip().upper()}"
    return CITY_COORDS.get(key_title)
