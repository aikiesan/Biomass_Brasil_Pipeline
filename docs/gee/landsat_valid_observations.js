// =============================================================================
// Cloud-cover proxy for bbp (H2): clear Landsat observations per municipality and year
// =============================================================================
// Paste into the Google Earth Engine Code Editor (code.earthengine.google.com) and Run.
// It starts one export task per year (Tasks tab -> Run each). Every task writes
// landsat_valid_obs_<year>.csv to the Google Drive folder below.
//
// What it measures: for every 30 m pixel, the number of Landsat 5/7/8/9 Collection 2
// Level-2 scenes in the calendar year whose QA_PIXEL flags it as clear (not fill,
// dilated cloud, cirrus, cloud, cloud shadow or snow). Per municipality it reports the
// mean of that count over the crop pixels of the MapBiomas coverage map of the same
// year (all pixels when CROP_MASK is false). MapBiomas classifies each year from these
// same scenes, so fewer clear observations = a harder classification.
//
// Before running (only step 1 needs an edit):
//  1. Upload the IBGE mesh (BR_Municipios_2025.shp + .dbf/.shx/.prj, zipped) as a table
//     asset: Assets -> NEW -> Shape files. Put its path in MUNICIPALITIES. It must keep
//     the column CD_MUN.
//  2. The crop mask comes from MapBiomas Collection 11 in the public collection
//     projects/mapbiomas-public/assets/brazil/lulc/v1 (one image per year, band `classification`,
//     properties collection_id and year; Collection 11 covers 1985–2025, checked 2026-10-15).
//     Set CROP_MASK = false to average over every pixel of the municipality instead.
//  3. Bring the CSVs back to
//     00_Primary_Data_Sources_PILAR2b/07_MapBiomas/Landsat_Valid_Obs/ (see pipeline.yaml).
// =============================================================================

var MUNICIPALITIES = 'projects/YOUR-PROJECT/assets/BR_Municipios_2025';   // <- edit
var MAPBIOMAS = ee.ImageCollection('projects/mapbiomas-public/assets/brazil/lulc/v1')
  .filter(ee.Filter.eq('collection_id', 11));
var CROP_MASK = true;
var CROP_CLASSES = [20, 39, 40, 41, 46, 47, 48, 62];   // MapBiomas Col. 11 temporary + perennial crops
var YEARS = ee.List.sequence(2008, 2024).getInfo();
var SCALE = 30;                 // native; 90 or 120 halves the run time with little change in a municipal mean
var DRIVE_FOLDER = 'bbp_landsat_valid_obs';

var municipalities = ee.FeatureCollection(MUNICIPALITIES).select(['CD_MUN']);

// QA_PIXEL bits (Collection 2): 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow
function clear(img) {
  var qa = img.select('QA_PIXEL');
  var bad = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5);
  return qa.bitwiseAnd(bad).eq(0).rename('clear').toUint8();
}

function landsat(year) {
  var start = ee.Date.fromYMD(year, 1, 1), end = start.advance(1, 'year');
  var ids = ['LANDSAT/LT05/C02/T1_L2', 'LANDSAT/LE07/C02/T1_L2', 'LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2'];
  var merged = ee.ImageCollection(ids[0]);
  for (var i = 1; i < ids.length; i++) merged = merged.merge(ee.ImageCollection(ids[i]));
  return merged.filterDate(start, end).filterBounds(municipalities.geometry());
}

YEARS.forEach(function (year) {
  var count = landsat(year).map(clear).sum().rename('clear_obs').unmask(0);
  var image = count;
  if (CROP_MASK) {
    var cov = MAPBIOMAS.filter(ee.Filter.eq('year', year)).first().select('classification');
    var crop = cov.remap(CROP_CLASSES, ee.List.repeat(1, CROP_CLASSES.length), 0).eq(1);
    image = count.updateMask(crop);
  }
  var stats = image.reduceRegions({
    collection: municipalities,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.count(), sharedInputs: true}),
    scale: SCALE,
    tileScale: 4
  }).map(function (f) {
    return ee.Feature(null, {CD_MUN: f.get('CD_MUN'), year: year, clear_obs_mean: f.get('mean'),
                             pixels: f.get('count'), crop_mask: CROP_MASK});
  });
  Export.table.toDrive({collection: stats, description: 'landsat_valid_obs_' + year, folder: DRIVE_FOLDER,
                        fileNamePrefix: 'landsat_valid_obs_' + year, fileFormat: 'CSV',
                        selectors: ['CD_MUN', 'year', 'clear_obs_mean', 'pixels', 'crop_mask']});
});
