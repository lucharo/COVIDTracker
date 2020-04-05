import pandas as pd
import json
from geopy.geocoders import Nominatim
import plotly.express as px
import plotly as py


geolocator = Nominatim(user_agent="")


def get_location(x):
    try:
        location = geolocator.geocode(x)
    except Exception:
        location = None
    if location is None:
        return (None, None)
    else:
        return (location.latitude, location.longitude)


def heatmap(stream_file=""):
    twitter_df = pd.read_json(stream_file, lines=True)
    json_struct = json.loads(twitter_df.to_json(orient="records"))
    df_flat = pd.json_normalize(json_struct)

    df_flat['latlon'] = df_flat['user.location'].apply(lambda x: get_location(x))
    df_flat[['lat', 'lon']] = pd.DataFrame(df_flat['latlon'].tolist(), index=df_flat.index)
    fig=px.density_mapbox(df_flat, lat="lat", lon="lon", radius=5, 
                     center=dict(lat=0, lon=180), zoom=0,
                     mapbox_style="open-street-map")
    py.offline.plot(fig,filename="test.html")
    # fig.show()
