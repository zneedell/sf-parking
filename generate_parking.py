
import pandas as pd
import geopandas as gpd
import statsmodels.formula.api as smf
import numpy as np
import osmnx as ox
import pickle
from scipy.optimize import least_squares
from shapely.geometry import  Point
import itertools

#%% Load TAZ data from MTC

taz = gpd.read_file('data/MTC/Transportation_Analysis_Zones.shp')
taz['taz_id'] = taz['taz1454']
taz = taz.set_index('taz1454', drop=True)
demo = pd.read_csv('data/MTC/Plan_Bay_Area_2040_Forecast__Population_and_Demographics.csv').set_index('zoneid',drop=True)
emp = pd.read_csv('data/MTC/Plan_Bay_Area_2040_Forecast__Employment.csv').set_index('zoneid',drop=True)
lut = pd.read_csv('data/MTC/Plan_Bay_Area_2040_Forecast__Land_Use_and_Transportation.csv').set_index('zoneid',drop=True)
taz = taz.merge(demo, left_index=True,right_index=True).merge(emp, left_index=True,right_index=True).merge(lut, left_index=True,right_index=True)
taz = pd.concat([taz.iloc[:,:8],taz.loc[:,taz.columns.str.endswith('15')]],axis=1)
taz['area'] = taz['geometry'].to_crs({'init': 'epsg:3395'}).area/10**3 # in 1000s of sq meters (good numerically)
sf_boundary = taz.loc[taz['county'] == 'San Francisco','geometry'].unary_union
cols = taz.columns
for col in cols[[8,20,21,22,23,24,25,26,27,31,32]]:
    taz.loc[taz[col] == 0,col] += 0.1
    taz[col+'_den'] = taz[col]/taz['area']

taz['AreaType'] = 'Residential'
taz.loc[taz['areatype15'] == 0,'AreaType'] = 'CBD'
taz.loc[taz['areatype15'] == 1,'AreaType'] = 'Downtown'
taz.loc[taz['areatype15'] == 2,'AreaType'] = 'Downtown'
taz.loc[taz['areatype15'] == 3,'AreaType'] = 'Residential'
taz['JobsAndResidents'] = taz['totemp15'] + taz['totpop15']
taz['JobsAndResidents_den'] = taz['JobsAndResidents'] / taz['area']

#%% Load BART distances

bart_distances = pd.read_csv('data/distanceToBart.csv')
taz = taz.merge(bart_distances,left_index=True,right_on='InputID')
taz.rename(columns={'Distance':'DistanceToBART'},inplace=True)
taz['NearBart'] = taz['DistanceToBART'] < 800

#%% Load OSM data and group road segments by TAZ

with open('gdfs.pickle', 'rb') as f:
    gdfs = pickle.load(f)

def center_of_linestring(x):
    return Point(np.mean(x.coords.xy[0]),np.mean(x.coords.xy[1]))
gdfs['geometry'] = gdfs['geometry'].apply(center_of_linestring)
gdfs['oneway'] = gdfs['oneway'].str == 'True'
gdfs.loc[~gdfs['oneway'],'length_corrected'] = gdfs.loc[~gdfs['oneway'],'length']/2
gdfs['classification'] = gdfs['highway'].str.replace('_link','').replace('living_street','unclassified')

gdfs.rename(columns={'length':'length_OSM','length_corrected':'length_corrected_OSM'},inplace=True)
OSM_with_taz = gpd.sjoin(taz,gdfs[['oneway','geometry','length_OSM','length_corrected_OSM','classification']],how='inner',op='contains')
taz_with_OSM = OSM_with_taz.groupby([OSM_with_taz.index,OSM_with_taz['classification']]).agg({'oneway':'sum','length_OSM':'sum','length_corrected_OSM':'sum'}).unstack(level=-1).fillna(0)
taz_with_OSM.columns = taz_with_OSM.columns.to_flat_index().map('_'.join)

#%% Group On Street Parking by TAZ

onstp = gpd.read_file('data/OnStreetParking/geo_export_9c00c3f8-0452-4427-add0-07e5c897479a.shp')
ECKERT_IV_PROJ4_STRING = "+proj=eck4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
onstp['length'] = onstp['geometry'].to_crs(ECKERT_IV_PROJ4_STRING).length
onstp['geometry'] = onstp['geometry'].apply(center_of_linestring)
onstp_with_taz = gpd.sjoin(taz,onstp[['prkg_sply','geometry','length']],how='inner',op='intersects')
taz_with_onstp = onstp_with_taz.groupby(onstp_with_taz.index).agg({'prkg_sply':'sum','length':'sum'}).fillna(0)
taz_with_onstp.rename(columns={'length':'length_SF','prkg_sply':'OnStreetParking'},inplace=True)

#%% Group Off Street Parking by TAZ
offstp = gpd.read_file('data/OffStreetParking/OSP_09162011.shp')
offstp['OffStreetParking'] = offstp['RegCap'] + offstp['ValetCap']
offstp['PaidPublicParking'] = offstp['OffStreetParking'] * (offstp['PrimeType'] == 'PPA')
offstp['FreePublicParking'] = offstp['OffStreetParking'] * ((offstp['PrimeType'] == 'FPA') | (offstp['PrimeType'] == 'CPO'))
offstp['WorkParking'] = offstp['OffStreetParking'] * ((offstp['PrimeType'] == 'PHO') | (offstp['PrimeType'] == 'CGO'))



offstp_with_taz = gpd.sjoin(taz,offstp[['RegCap','ValetCap','MCCap','OffStreetParking','PaidPublicParking','FreePublicParking','WorkParking','geometry']],how='inner',op='intersects')
taz_with_offstp = offstp_with_taz.groupby(offstp_with_taz.index).agg({'objectid':'first','OffStreetParking':'sum','PaidPublicParking':'sum','FreePublicParking':'sum','WorkParking':'sum'}).fillna(0)

#%% Group Parking Meters by TAZ

parking_meters = pd.read_csv('data/OnStreetParking/Parking_Meters.csv')
parking_meters = parking_meters.loc[parking_meters['ON_OFFSTREET_TYPE']=='ON',:]
parking_meters = gpd.GeoDataFrame(parking_meters,geometry=gpd.points_from_xy(parking_meters['LONGITUDE'],parking_meters['LATITUDE']))
parking_meters.crs = {'init': 'epsg:4326'}
meters_with_taz = gpd.sjoin(taz,parking_meters[['OBJECTID','geometry']],how='inner',op='intersects')
taz_with_meters = meters_with_taz.groupby(meters_with_taz.index).agg({'OBJECTID':'count'}).fillna(0)
taz_with_meters.rename(columns={'OBJECTID':'ParkingMeters'},inplace=True)


#%%
taz_all = taz.merge(taz_with_onstp,left_index=True,right_index=True,how='left')
taz_all = taz_all.merge(taz_with_OSM,left_index=True,right_index=True,how='left')
taz_all = taz_all.merge(taz_with_offstp,left_index=True,right_index=True,how='left')
taz_all = taz_all.merge(taz_with_meters, left_index=True, right_index = True, how='left')


taz_all['AllParking'] = (taz_all['OffStreetParking'] + taz_all['OnStreetParking'])
taz_all['AllParking_den'] = taz_all['AllParking'] / taz_all['area']
taz_all['OffStreetParking_den'] = taz_all['OffStreetParking']/ taz_all['area']
taz_all['OnStreetParking_den'] = taz_all['OnStreetParking']/ taz_all['area']
taz_all['PortionOnStreet'] = taz_all['OnStreetParking'] / (taz_all['AllParking'] + 0.1 )
taz_all['PortionOnStreetPaid'] = taz_all['ParkingMeters'] / (taz_all['OnStreetParking'] + 0.1)
taz_all['PortionOffStreetPaid'] = taz_all['PaidPublicParking'] / (taz_all['OffStreetParking'] + 0.1)
#taz_all.loc[taz_all['PortionOffStreetPaid'] > 1,'PortionOffStreetPaid'] = 1 # Why do these exist?
taz_all.loc[taz_all['PortionOnStreetPaid'] > 1, 'PortionOnStreetPaid'] = 1
taz_all.fillna(0,inplace=True)
#%% Get Just SF data to train models

sf = taz_all.loc[taz['county']=='San Francisco',:]
sf['OnStreetParkingPerDistance'] = sf['OnStreetParking']/sf['length_SF']
sf['JobsPerOffStreetParking'] = sf['totemp15']/ (sf['OffStreetParking'] + 1)
sf['JobsPerOnStreetParking'] = sf['totemp15']/ (sf['OnStreetParking'] + 1)
sf['OnStreetParkingPerJob'] = sf['OnStreetParking'] /(sf['totemp15'] + 1)
sf['OffStreetParkingPerJob'] = sf['OffStreetParking'] /(sf['totemp15'] + 1)


sf = sf.loc[sf.index != 1258,:]# Get rid of candlestick park (outlier high off street parking)
sf = sf.loc[sf.index != 1074,:]# Get rid of treasure island (outlier low on street parking, not measured maybe)

#%% Train On Steet Parking Model

onstreet_simple = smf.ols(formula = 'OnStreetParking ~  -1 + length_OSM_primary + length_OSM_residential + length_OSM_secondary +length_OSM_trunk', data = sf)
onstreet_simple_res = onstreet_simple.fit()
print(onstreet_simple_res.summary())

# TODO: Add back in motorway once there is internet
onstreet_full = smf.ols(formula = 'OnStreetParking ~  -1 + length_OSM_tertiary+ length_OSM_primary + length_OSM_residential + length_OSM_secondary +length_OSM_trunk', data = sf)
onstreet_full_res = onstreet_full.fit()
print(onstreet_full_res.summary())

#%% Train On Street Paid Parking Model

metered_simple = smf.logit(formula = 'PortionOnStreetPaid ~ np.log(totemp15_den)', data = sf)
metered_simple_res = metered_simple.fit()
print(metered_simple_res.summary())

metered_full = smf.logit(formula = 'PortionOnStreetPaid ~ np.log(totemp15_den):C(AreaType) + np.log(sfdu15_den) + np.log(mfdu15_den)', data = sf)
metered_full_res = metered_full.fit()
print(metered_full_res.summary())

#%% Train Off Street Model

def fun(params, inputs, output):
    return np.sum(inputs[:,:-2]*params[:-3],axis=1)/(1 + np.exp(- params[-3]+ params[-2]*inputs[:,-2]  + params[-1]*inputs[:,-1] )) - output

inputs = np.vstack([sf['retempn15_den'],sf['fpsempn15_den'],sf['herempn15_den'],sf['othempn15_den'], sf['mfdu15_den'], sf['sfdu15_den'], np.log(sf['totemp15_den']), np.log(sf['totpop15_den'])]).transpose()
outputs = sf['OffStreetParking_den'].values
x0 = np.ones(9)

off_street_full = least_squares(fun, x0, loss='linear', f_scale=5, args=(inputs, outputs))
print('Complex Model Params', off_street_full.x)
print('Complex Model Cost', off_street_full.cost)

def fun_simple(params, inputs, output):
    return np.sum(inputs[:,:-2]*params[:-3],axis=1)/(1 + np.exp(- params[-3] + params[-2]*inputs[:,-2]  + params[-1]*inputs[:,-1] )) - output

inputs = np.vstack([sf['totemp15_den'],np.log(sf['totemp15_den']), np.log(sf['totpop15_den'])]).transpose()
outputs = sf['OffStreetParking_den'].values
x0 = np.ones(4)

off_street_simple = least_squares(fun_simple, x0, loss='linear', f_scale=5, args=(inputs, outputs))
print('Simple Model Params', off_street_simple.x)
print('Simple Model Cost', off_street_simple.cost)



#%% Train Off Street Paid Model
off_street_paid_full = smf.logit(formula = 'PortionOffStreetPaid ~ C(AreaType) + np.log(retempn15_den) + np.log(fpsempn15_den) + np.log(herempn15_den) + np.log(mwtempn15_den) + np.log(othempn15_den) + np.log(sfdu15_den):C(AreaType) + np.log(mfdu15_den)', data = sf)
off_street_paid_full_res = off_street_paid_full.fit()
print(off_street_paid_full_res.summary())


off_street_paid_simple = smf.logit(formula = 'PortionOffStreetPaid ~ np.log(totemp15_den) + np.log(totpop15_den) ', data = sf)
off_street_paid_res = off_street_paid_simple.fit()
print(off_street_paid_res.summary())

#%% Train Short Term Hourly Rate Model

st_parking_cost = smf.ols(formula = 'oprkcst15 ~  np.log(totemp15_den) + np.log(totpop15_den)', data = sf)
st_parking_cost_res = st_parking_cost.fit()
print(st_parking_cost_res.summary())

#%% Train Long Term Hourly Rate Model

lt_parking_cost = smf.ols(formula = 'prkcst15 ~  np.log(totemp15_den) + np.log(totpop15_den)', data = sf)
lt_parking_cost_res = lt_parking_cost.fit()
print(lt_parking_cost_res.summary())

#%% Fit Models for SFBAY

# On Street
taz_all['PredictedOnStreetParking'] = onstreet_full_res.predict(taz_all)

# On Street Paid
taz_all['PredictedOnStreetPortionPaid'] = metered_full_res.predict(taz_all)
taz_all['PredictedOnStreetPaidParking'] = np.ceil(taz_all['PredictedOnStreetParking'] * taz_all['PredictedOnStreetPortionPaid'])
taz_all['PredictedOnStreetFreeParking'] = np.ceil(taz_all['PredictedOnStreetParking'] - taz_all['PredictedOnStreetPaidParking'])

# Off Street
def predict_off_street(params, inputs):
    return np.sum(inputs[:,:-2]*params[:-3],axis=1)/(1 + np.exp(- params[-3]+ params[-2]*inputs[:,-2]  + params[-1]*inputs[:,-1] ))

full_inputs = np.vstack([taz_all['retempn15_den'],taz_all['fpsempn15_den'],taz_all['herempn15_den'],taz_all['othempn15_den'], taz_all['mfdu15_den'], taz_all['sfdu15_den'], np.log(taz_all['totemp15_den']), np.log(taz_all['totpop15_den'])]).transpose()
taz_all['PredictedOffStreetParking'] = predict_off_street(off_street_full.x,full_inputs) * taz_all['area']
taz_all.loc[taz_all['PredictedOffStreetParking'] < 0,'PredictedOffStreetParking'] = 0
taz_all['PredictedOffStreetParking_extra'] = taz_all['PredictedOffStreetParking'] + 0.25*taz_all['mwtempn15'] + 0.25*taz_all['agrempn15'] # Make sure manufacturing and ag jobs have places to park (not well represented in SF)

# Off Street Paid
taz_all['PredictedOffStreetPortionPaid'] = off_street_paid_full_res.predict(taz_all)
taz_all['PredictedOffStreetPaidParking'] = np.ceil(taz_all['PredictedOffStreetParking'] * taz_all['PredictedOffStreetPortionPaid'])
taz_all['PredictedOffStreetFreeParking'] = np.ceil(taz_all['PredictedOffStreetParking'] - taz_all['PredictedOffStreetPaidParking'])

# =============================================================================
# # Costs: To compare predictions from just SF to inputs from entire bay area
# taz_all['ShortTermHourlyRate'] = st_parking_cost_res.predict(taz_all)
# taz_all.loc[taz_all['ShortTermHourlyRate'] < 0, 'ShortTermHourlyRate'] = 0
# 
# taz_all['LongTermHourlyRate'] = lt_parking_cost_res.predict(taz_all)
# taz_all.loc[taz_all['LongTermHourlyRate'] < 0, 'LongTermHourlyRate'] = 0
#
# sns.scatterplot(x='LongTermHourlyRate', y='prkcst15', hue='AreaType', data=taz_all)
# =============================================================================
taz_all.loc[taz_all['areatype15'] == 4,'AreaType'] = 'Suburbs' # None of these in SF so they screw up the model


taz_all['MadeUpWorkSpaces'] = 0 # If no area type available, add when employment density is greater than 2?
taz_all.loc[taz_all['AreaType'] == 'Suburbs','MadeUpWorkSpaces'] = taz_all['totemp15']
taz_all.loc[taz_all['AreaType'] == 'Residential','MadeUpWorkSpaces'] = taz_all['totemp15']

taz_all['MadeUpResidentialSpaces'] = 0
taz_all.loc[taz_all['AreaType'] == 'Suburbs','MadeUpResidentialSpaces'] = np.ceil(3*(taz_all['sfdu15'] + taz_all['mfdu15']))
taz_all.loc[taz_all['AreaType'] == 'Residential','MadeUpResidentialSpaces'] = np.ceil(1.5*taz_all['sfdu15'] + 0.75*taz_all['mfdu15'])
taz_all.loc[taz_all['AreaType'] == 'Downtown','MadeUpResidentialSpaces'] = np.ceil(taz_all['sfdu15'] + 0.5*taz_all['mfdu15'])
taz_all.loc[taz_all['AreaType'] == 'CBD','MadeUpResidentialSpaces'] = np.ceil(0.5*taz_all['sfdu15'] + 0.25*taz_all['mfdu15'])
#%% Build output file

output = []



for idx, row in taz_all.iterrows():
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':row.PredictedOnStreetPaidParking,'feeInCents':row.oprkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':row.PredictedOnStreetFreeParking,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':row.PredictedOffStreetPaidParking,'feeInCents':row.prkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':row.PredictedOffStreetFreeParking,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Residential','pricingModel':'Block','chargingType':'Level2','numStalls':row.MadeUpResidentialSpaces,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Workplace','pricingModel':'Block','chargingType':'Level2','numStalls':row.MadeUpWorkSpaces,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
output = pd.DataFrame(output, columns = ['taz','parkingType','pricingModel','chargingType','numStalls','feeInCents','ReservedFor'])
output = output.loc[output['numStalls'] > 0, :]
output.to_csv('output/taz-parking-base.csv',index=False)



#%% Build output file


taz_all.loc[:,'PredictedOffStreetPaidParking'] = np.floor(taz_all.loc[:,'PredictedOffStreetPaidParking']*.75)
taz_all.loc[:,'PredictedOffStreetFreeParking'] = np.floor(taz_all.loc[:,'PredictedOffStreetFreeParking']*.5)
taz_all.loc[:,'oprkcst15'] = np.floor(taz_all.loc[:,'oprkcst15']*2)
taz_all.loc[:,'prkcst15'] = np.floor(taz_all.loc[:,'prkcst15']*2)


output = []



for idx, row in taz_all.iterrows():
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':row.PredictedOnStreetPaidParking,'feeInCents':row.oprkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':row.PredictedOnStreetFreeParking,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':row.PredictedOffStreetPaidParking,'feeInCents':row.prkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':row.PredictedOffStreetFreeParking,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Residential','pricingModel':'Block','chargingType':'Level2','numStalls':row.MadeUpResidentialSpaces,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz_id,'parkingType':'Workplace','pricingModel':'Block','chargingType':'Level2','numStalls':row.MadeUpWorkSpaces,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
output = pd.DataFrame(output, columns = ['taz','parkingType','pricingModel','chargingType','numStalls','feeInCents','ReservedFor'])
output = output.loc[output['numStalls'] > 0, :]
output.to_csv('output/taz-parking-low.csv',index=False)

#%% Do the same thing for sf-light tazs

taz_sflight = gpd.read_file('data/BEAM/taz/sf-light-tazs.shp').to_crs({'init': 'epsg:4326'})
taz_sflight['taz_id'] = taz_sflight['name']
taz_sflight = taz_sflight.set_index('name', drop=True)
onstp_with_sftaz = gpd.sjoin(taz_sflight,onstp[['prkg_sply','geometry','length']],how='inner',op='intersects')
sftaz_with_onstp = onstp_with_sftaz.groupby(onstp_with_sftaz.index).agg({'prkg_sply':'sum','length':'sum'}).fillna(0)
sftaz_with_onstp.rename(columns={'length':'length_SF','prkg_sply':'OnStreetParking'},inplace=True)

offstp_with_sftaz = gpd.sjoin(taz_sflight,offstp[['RegCap','ValetCap','MCCap','OffStreetParking','PaidPublicParking','FreePublicParking','WorkParking','geometry']],how='inner',op='intersects')
sftaz_with_offstp = offstp_with_sftaz.groupby(offstp_with_sftaz.index).agg({'taz_id':'first','OffStreetParking':'sum','PaidPublicParking':'sum','FreePublicParking':'sum','WorkParking':'sum'}).fillna(0)


meters_with_sftaz = gpd.sjoin(taz_sflight,parking_meters[['OBJECTID','geometry']],how='inner',op='intersects')
sftaz_with_meters = meters_with_sftaz.groupby(meters_with_sftaz.index).agg({'OBJECTID':'count'}).fillna(0)
sftaz_with_meters.rename(columns={'OBJECTID':'ParkingMeters'},inplace=True)


sftaz_with_baytaz = gpd.sjoin(gpd.GeoDataFrame(taz_sflight['geometry'].centroid).rename(columns={0:'geometry'}).set_geometry('geometry'), taz_all[['geometry','oprkcst15','prkcst15','taz_id','sfdu15_den','mfdu15_den','totpop15_den']], how='left').rename(columns={'geometry':'centroid'})

sftaz_all = taz_sflight.merge(sftaz_with_onstp,left_index=True,right_index=True,how='left')
sftaz_all = sftaz_all.merge(sftaz_with_offstp,left_index=True,right_index=True,how='left')
sftaz_all = sftaz_all.merge(sftaz_with_meters, left_index=True, right_index = True, how='left')
sftaz_all = sftaz_all.merge(sftaz_with_baytaz, left_index=True, right_index = True, how='left').fillna(0)
sftaz_all['inSFproper'] = sftaz_all['geometry'].apply(lambda x: sf_boundary.contains(x.centroid))
sftaz_all['area'] = sftaz_all['geometry'].to_crs({'init': 'epsg:3395'}).area/10**3 # in 1000s of sq meters (good numerically)
sftaz_all['sfdu15'] = sftaz_all['sfdu15_den'] * sftaz_all['area']
sftaz_all['mfdu15'] = sftaz_all['mfdu15_den'] * sftaz_all['area']
sftaz_all['totpop15'] = sftaz_all['totpop15_den'] * sftaz_all['area']
sftaz_all['MadeUpResidentialSpaces'] = np.ceil(sftaz_all['sfdu15'] + 0.5*sftaz_all['mfdu15'])

sftaz_all.loc[~sftaz_all['inSFproper'],['OffStreetParking','FreePublicParking','WorkParking','OnStreetParking']] = 1000000


#%% Build output file

output = []



for idx, row in sftaz_all.iterrows():
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':row.ParkingMeters,'feeInCents':row.oprkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':np.min([row.OnStreetParking - row.ParkingMeters,0]),'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':row.PaidPublicParking,'feeInCents':row.prkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':row.FreePublicParking,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Residential','pricingModel':'Block','chargingType':'Level2','numStalls':row.MadeUpResidentialSpaces,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Workplace','pricingModel':'Block','chargingType':'Level2','numStalls':row.WorkParking,'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
output = pd.DataFrame(output, columns = ['taz','parkingType','pricingModel','chargingType','numStalls','feeInCents','ReservedFor'])
output = output.loc[output['numStalls'] > 0, :]
output.to_csv('output/sf-taz-parking-base.csv',index=False)


#%% Build output file

output = []

sf_pop = sftaz_all['totpop15'].sum()
simulated_pop = 2500
factor = simulated_pop/sf_pop

for idx, row in sftaz_all.iterrows():
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':np.ceil(row.ParkingMeters*factor),'feeInCents':row.oprkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'NoCharger','numStalls':np.ceil(np.min([row.OnStreetParking - row.ParkingMeters,0])*factor),'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':np.ceil(row.PaidPublicParking*factor),'feeInCents':row.prkcst15,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Public','pricingModel':'Block','chargingType':'Level2','numStalls':np.ceil(row.FreePublicParking*factor),'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Residential','pricingModel':'Block','chargingType':'Level2','numStalls':np.ceil(row.MadeUpResidentialSpaces*factor),'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
    newrow = {'taz':row.taz,'parkingType':'Workplace','pricingModel':'Block','chargingType':'Level2','numStalls':np.ceil(row.WorkParking*factor),'feeInCents':0,'ReservedFor':'Any'}
    output.append(newrow)
output = pd.DataFrame(output, columns = ['taz','parkingType','pricingModel','chargingType','numStalls','feeInCents','ReservedFor'])
output = output.loc[output['numStalls'] > 0, :]
output['numStalls'] = output['numStalls'].astype('int')
output.to_csv('output/sf-taz-parking-base-2.5k.csv',index=False)