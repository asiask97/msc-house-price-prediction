# England House Price Data Pipeline

### Land Registry Features

| # | Column | Description |
|---|--------|-------------|
| 1 | Transaction ID | Unique auto-generated reference (the UUID you see) |
| 2 | Price | Sale price on the transfer deed |
| 3 | Date | Date sale completed |
| 4 | Postcode | Postcode at time of transaction |
| 5 | Property Type | D=Detached, S=Semi-detached, T=Terraced, F=Flat, O=Other |
| 6 | New Build | Y=New build, N=Not new build |
| 7 | Duration | F=Freehold, L=Leasehold |
| 8 | PAON | Primary Addressable Object Name (house number/name) |
| 9 | SAON | Secondary Addressable Object Name (flat/unit within building) |
| 10 | Street | |
| 11 | Locality | |
| 12 | Town/City | |
| 13 | District | |
| 14 | County | |
| 15 | PPD Category | A=Standard, B=Additional (repossessions, buy-to-lets) |
| 16 | Record Status | A=Addition, C=Change, D=Delete |

Source: https://www.gov.uk/guidance/about-the-price-paid-data

---

### ONS Postcode Directory Features

<details>
<summary>All ONSPD Fields</summary>

| # | Field | Description |
|---|-------|-------------|
| 1 | OBJECTID | ArcGIS object identifier |
| 2 | PCD7 | Unit postcode – 7 character version |
| 3 | PCD8 | Unit postcode – 8 character version |
| 4 | PCDS | Unit postcode – variable length (eGif) version |
| 5 | DOINTR | Date of introduction |
| 6 | DOTERM | Date of termination (null if live) |
| 7 | CTY25CD | County (2025) |
| 8 | CED25CD | County Electoral Division (2025) |
| 9 | LAD25CD | Local Authority District (2025) |
| 10 | WD25CD | Electoral Ward (2025) |
| 11 | PARNCP25CD | Parish / Community (2025) |
| 12 | USRTYPIND | Postcode user type (0=small, 1=large) |
| 13 | EAST1M | National grid reference – Easting |
| 14 | NORTH1M | National grid reference – Northing |
| 15 | GRIDIND | Grid reference positional quality indicator |
| 16 | HLTH19CD | Former Strategic Health Authority / Health Board (2019) |
| 17 | NHSER24CD | NHS England Region (2024) |
| 18 | CTRY25CD | Country (2025) |
| 19 | RGN25CD | Region (2025) |
| 20 | SSR95CD | Standard Statistical Region (1995) |
| 21 | PCON24CD | Westminster Parliamentary Constituency (2024) |
| 22 | EER20CD | European Electoral Region (2020) |
| 23 | EDUC23CD | Local Learning and Skills Council / Enterprise Region (2023) |
| 24 | TTWA15CD | Travel to Work Area (2015) |
| 25 | PCO19CD | Primary Care Organisation (2019) |
| 26 | ITL25CD | International Territorial Level (2025) |
| 27 | WDSTL05CD | 2005 Statistical Ward |
| 28 | OA01CD | 2001 Census Output Area |
| 29 | WDCAS03CD | 2003 Census Area Statistics Ward |
| 30 | NPARK16CD | National Park (2016) |
| 31 | LSOA01CD | 2001 Census Lower Layer Super Output Area |
| 32 | MSOA01CD | 2001 Census Middle Layer Super Output Area |
| 33 | RUC01IND | 2001 Census rural-urban classification |
| 34 | OAC01IND | 2001 Census Output Area classification |
| 35 | OA11CD | 2011 Census Output Area |
| 36 | LSOA11CD | 2011 Census Lower Layer Super Output Area |
| 37 | MSOA11CD | 2011 Census Middle Layer Super Output Area |
| 38 | WZ11CD | 2011 Census Workplace Zone |
| 39 | SICBL24CD | Sub ICB Location (2024) |
| 40 | BUA24CD | Built-up Area (2024) |
| 41 | RUC11IND | 2011 Census rural-urban classification |
| 42 | OAC11IND | 2011 Census Output Area classification |
| 43 | LAT | Decimal degrees latitude |
| 44 | LONG | Decimal degrees longitude |
| 45 | LEP21CD1 | Local Enterprise Partnership – first instance (2021) |
| 46 | LEP21CD2 | Local Enterprise Partnership – second instance (2021) |
| 47 | PFA23CD | Police Force Area (2023) |
| 48 | IMD20IND | Index of Multiple Deprivation (2020) |
| 49 | CAL24CD | Cancer Alliance (2024) |
| 50 | ICB23CD | Integrated Care Board (2023) |
| 51 | OA21CD | 2021 Census Output Area |
| 52 | LSOA21CD | 2021 Census Lower Layer Super Output Area |
| 53 | MSOA21CD | 2021 Census Middle Layer Super Output Area |
| 54 | RUC21IND | 2021 Census rural-urban classification |

</details>

Fields used: `PCDS`, `LAT`, `LONG` — postcode to latitude/longitude lookup only.

---

### EPC Features Used

Source: https://epc.opendatacommunities.org/docs/guidance

> **Note:** `loading_data.py` must be run first to load EPC data into a parquet file, which is much faster to read in notebooks. It also removes any EPC certificates whose postcodes are not present in the Land Registry dataset.

| Column | Description |
|--------|-------------|
| postcode | Property postcode |
| lodgement_date | Date EPC was lodged on register |
| total_floor_area | Floor area m² |
| number_habitable_rooms | Number of habitable rooms |
| current_energy_rating | A-G rating |
| current_energy_efficiency | Numeric efficiency score |
| construction_age_band | Age band e.g. 1967-1975 |
| built_form | Detached, Semi-Detached, Terrace etc |
| property_type | House, Flat, Bungalow etc |
| main_fuel | Gas, electricity etc |
| mains_gas_flag | Y/N |
| transaction_type | What triggered the EPC e.g. marketed sale |
| tenure | Owner-occupied, rented etc |

> **Important:** EPC data only exists from 2008 when certificates became mandatory for property sales. Any Land Registry sale before 2008 will have no EPC match and EPC columns will be `NaN`. These records are kept in the final dataset but have no EPC features.


#### EPC Timing

We don't know if someone completed the EPC certificate before the house was sold or after. Only EPCs lodged **before or on the sale date** are used — post-sale EPCs are discarded to prevent the model seeing information that wouldn't have been available at time of sale.

---
### UPRN

A UPRN (Unique Property Reference Number) is a permanent numeric ID assigned by Ordnance Survey to every addressable location in the UK, used to uniquely identify a property regardless of how its address is formatted.

<details>
<summary>OpenUPRN Fields</summary>

| # | Field | Description |
|---|-------|-------------|
| 1 | UPRN | Unique Property Reference Number — permanent ID assigned to every addressable location in the UK |
| 2 | X_COORDINATE | Ordnance Survey easting (British National Grid) |
| 3 | Y_COORDINATE | Ordnance Survey northing (British National Grid) |
| 4 | LATITUDE | Decimal degrees latitude (WGS84) |
| 5 | LONGITUDE | Decimal degrees longitude (WGS84) |

Source: https://osdatahub.os.uk/downloads/open/OpenUPRN

</details>

---

## Running the Pipeline

Run in this order:

```bash
python loading_data.py          # once only — loads and saves EPC as parquet
python joining_epc_and_lr.py    # matches LR sales to EPC records
python add_exact_coordinates.py # adds postcode geo-location or exact when possible
python add_school_distances.py  # adds distances to primary and secondary schools
```

### `loading_data.py`
Loads all EPC CSVs, filters to postcodes in the Land Registry, and saves as `Data/EPC/epc.parquet`. Run once — after this the notebook loads EPC in seconds instead of minutes.

### `joining_epc_and_lr.py`
Matches each Land Registry sale to its closest EPC before the sale date using postcode + normalised address. Outputs `Data/lr_epc_matched.parquet` — one row per sale with EPC columns appended. Unmatched sales are included with `NaN` in EPC columns.

```
**Match results:** 
--- Results ---
Total LR sales:       5,890,089
Matched to EPC:       4,379,248 (74.3%)
Unmatched:            1,510,841 (25.7%)
```

### `add_exact_coordinates.py`
Enriches the matched dataset with precise per-property coordinates from OpenUPRN, where a UPRN is available in the EPC data. Falls back to postcode centroid from ONS Postcode Directory where no UPRN exists. Adds `exact_lat`, `exact_lon` and `has_exact_coords` columns. Outputs `Data/lr_epc_coords.parquet`.

```
**Match results:** 
--- Results ---
Total records:              5,890,089
Exact coords (OpenUPRN):    4,371,593 (74.2%)
Centroid only (ONSPD):      1,495,460
No coordinates at all:      23,036
```

---

### `add_school_distances.py`
Calculates distance to nearest primary and nearest secondary school for each property. School locations come from two sources: GIAS for England and DataMapWales for Wales. Both use British National Grid coordinates which are converted to lat/long before distance calculation. Uses BallTree for efficient nearest-neighbour lookup across ~5.9M properties. Outputs `Data/lr_epc_schools.parquet`.

```
**Match results:**
--- Distance Summary ---
Primary: mean=0.57km, median=0.45km, max=51.13km
Secondary: mean=1.73km, median=1.13km, max=53.47km
--- Joined amounts ---
Primary schools:   17,866
Secondary schools: 3,339
```

---

## Spital Features

### Rural/Urban Classification (RUC21IND) 
from Online_ONS_Postcode_Directory_Live.csv. Not added right now because model feature since distance features will capture the same information. Will be used to split results by urban/rural during evaluation.

### Distance from Schools
Distance to nearest primary and nearest secondary school for each property. School locations for England are from GIAS dataset and Wales from https://datamap.gov.wales/ . Distance calculated from property coordinates to nearest school using lat/long. 

### School Ofsted Rating
Not included for now. Only covers England, changes every few years, and distance alone should capture most of the effect. Could add later if needed.

---

## Spital Features Datasets

### Schools - England GIAS
Source - https://get-information-schools.service.gov.uk/Downloads 

Fields used: `EstablishmentName`, `EstablishmentStatus (name)`, `PhaseOfEducation (name)`, `Easting`, `Northing`
 
<details>
<summary>All England GIAS Fields</summary>

| # | Field | Description |
|---|-------|-------------|
| 1 | URN | Unique Reference Number for the school |
| 2 | LA (code) | Local authority code |
| 3 | LA (name) | Local authority name |
| 4 | EstablishmentNumber | School establishment number |
| 5 | EstablishmentName | Name of the school |
| 6 | TypeOfEstablishment (code) | |
| 7 | TypeOfEstablishment (name) | e.g. Community school, Academy |
| 8 | EstablishmentTypeGroup (code) | |
| 9 | EstablishmentTypeGroup (name) | e.g. Local authority maintained, Independent |
| 10 | EstablishmentStatus (code) | |
| 11 | EstablishmentStatus (name) | Open, Closed, etc. |
| 12 | ReasonEstablishmentOpened (code) | |
| 13 | ReasonEstablishmentOpened (name) | |
| 14 | OpenDate | Date school opened |
| 15 | ReasonEstablishmentClosed (code) | |
| 16 | ReasonEstablishmentClosed (name) | |
| 17 | CloseDate | Date school closed |
| 18 | PhaseOfEducation (code) | |
| 19 | PhaseOfEducation (name) | Primary, Secondary, Nursery, etc. |
| 20 | StatutoryLowAge | Lowest age of pupils |
| 21 | StatutoryHighAge | Highest age of pupils |
| 22 | Boarders (code) | |
| 23 | Boarders (name) | Whether school has boarders |
| 24 | NurseryProvision (name) | Whether school has nursery classes |
| 25 | OfficialSixthForm (code) | |
| 26 | OfficialSixthForm (name) | Whether school has a sixth form |
| 27 | Gender (code) | |
| 28 | Gender (name) | Boys, Girls, Mixed |
| 29 | ReligiousCharacter (code) | |
| 30 | ReligiousCharacter (name) | e.g. Church of England, None |
| 31 | ReligiousEthos (name) | |
| 32 | Diocese (code) | |
| 33 | Diocese (name) | |
| 34 | AdmissionsPolicy (code) | |
| 35 | AdmissionsPolicy (name) | e.g. Selective, Not applicable |
| 36 | SchoolCapacity | Maximum number of pupils |
| 37 | SpecialClasses (code) | |
| 38 | SpecialClasses (name) | |
| 39 | CensusDate | Date of last school census |
| 40 | NumberOfPupils | Total pupils on roll |
| 41 | NumberOfBoys | |
| 42 | NumberOfGirls | |
| 43 | PercentageFSM | Percentage eligible for free school meals |
| 44 | TrustSchoolFlag (code) | |
| 45 | TrustSchoolFlag (name) | |
| 46 | Trusts (code) | |
| 47 | Trusts (name) | |
| 48 | SchoolSponsorFlag (name) | |
| 49 | SchoolSponsors (name) | |
| 50 | FederationFlag (name) | |
| 51 | Federations (code) | |
| 52 | Federations (name) | |
| 53 | UKPRN | UK Provider Reference Number |
| 54 | FEHEIdentifier | |
| 55 | FurtherEducationType (name) | |
| 56 | LastChangedDate | |
| 57 | Street | |
| 58 | Locality | |
| 59 | Address3 | |
| 60 | Town | |
| 61 | County (name) | |
| 62 | Postcode | |
| 63 | SchoolWebsite | |
| 64 | TelephoneNum | |
| 65 | HeadTitle (name) | |
| 66 | HeadFirstName | |
| 67 | HeadLastName | |
| 68 | HeadPreferredJobTitle | |
| 69 | BSOInspectorateName (name) | |
| 70 | InspectorateReport | |
| 71 | DateOfLastInspectionVisit | |
| 72 | NextInspectionVisit | |
| 73 | TeenMoth (name) | |
| 74 | TeenMothPlaces | |
| 75 | CCF (name) | |
| 76 | SENPRU (name) | |
| 77 | EBD (name) | |
| 78 | PlacesPRU | |
| 79 | FTProv (name) | |
| 80 | EdByOther (name) | |
| 81 | Section41Approved (name) | |
| 82 | SEN1–SEN13 (name) | Special educational needs categories |
| 83 | TypeOfResourcedProvision (name) | |
| 84 | ResourcedProvisionOnRoll | |
| 85 | ResourcedProvisionCapacity | |
| 86 | SenUnitOnRoll | |
| 87 | SenUnitCapacity | |
| 88 | GOR (code) | |
| 89 | GOR (name) | Government Office Region |
| 90 | DistrictAdministrative (code) | |
| 91 | DistrictAdministrative (name) | |
| 92 | AdministrativeWard (code) | |
| 93 | AdministrativeWard (name) | |
| 94 | ParliamentaryConstituency (code) | |
| 95 | ParliamentaryConstituency (name) | |
| 96 | UrbanRural (code) | |
| 97 | UrbanRural (name) | Urban/rural classification |
| 98 | GSSLACode (name) | |
| 99 | Easting | British National Grid easting |
| 100 | Northing | British National Grid northing |
| 101 | MSOA (name) | Middle Layer Super Output Area |
| 102 | LSOA (name) | Lower Layer Super Output Area |
| 103 | InspectorateName (name) | e.g. Ofsted, ISI |
| 104 | SENStat | |
| 105 | SENNoStat | |
| 106 | BoardingEstablishment (name) | |
| 107 | PropsName | |
| 108 | PreviousLA (code) | |
| 109 | PreviousLA (name) | |
| 110 | PreviousEstablishmentNumber | |
| 111 | Country (name) | |
| 112 | UPRN | Unique Property Reference Number |
| 113 | SiteName | |
| 114 | QABName (code) | |
| 115 | QABName (name) | |
| 116 | EstablishmentAccredited (code) | |
| 117 | EstablishmentAccredited (name) | |
| 118 | QABReport | |
| 119 | CHNumber | Companies House number |
| 120 | MSOA (code) | |
| 121 | LSOA (code) | |
| 122 | FSM | Free school meals |
| 123 | AccreditationExpiryDate | |
 
</details>

---

### Schools - Wales 
Source - https://datamap.gov.wales/layers/geonode:maintained_schools_wg

Fields used: `school_name`, `sector`, `geom`
 
> **Note:** Welsh dataset uses Welsh language labels for sector: Cynradd = Primary, Uwchradd = Secondary, Arbennig = Special, Canol = Middle, Meithrin = Nursery. Both datasets use British National Grid coordinates (Easting/Northing) which are converted to WGS84 lat/long using pyproj.
 
<details>
<summary>All Wales DataMapWales Fields</summary>

| # | Field | Description |
|---|-------|-------------|
| 1 | FID | Feature ID |
| 2 | uprn | Unique Property Reference Number |
| 3 | school_code | |
| 4 | school_name | Name of the school |
| 5 | la_code | Local authority code |
| 6 | local_authority | Local authority name |
| 7 | sector | School phase (Cynradd/Uwchradd/Arbennig/Canol/Meithrin) |
| 8 | governance | e.g. Community |
| 9 | wm_code | Welsh medium code |
| 10 | welsh_medium | Welsh medium type |
| 11 | school_type | e.g. Nursery, Infants & Juniors |
| 12 | religious_character | |
| 13 | address_1 | |
| 14 | address_2 | |
| 15 | address_3 | |
| 16 | address_4 | |
| 17 | postcode | |
| 18 | phone_number | |
| 19 | pupils | Number of pupils |
| 20 | rhif_yr_ysgol | School number (Welsh label) |
| 21 | enwr_ysgol | School name (Welsh label) |
| 22 | cod_all | LA code (Welsh label) |
| 23 | awdurdod_lleol | Local authority (Welsh label) |
| 24 | llywodraethu | Governance (Welsh label) |
| 25 | cod_cc | Welsh medium code (Welsh label) |
| 26 | math_o_gyfrwng_cymraeg | Welsh medium type (Welsh label) |
| 27 | math_o_ysgol | School type (Welsh label) |
| 28 | cymeriad_crefyddol | Religious character (Welsh label) |
| 29 | cyfeiriad_1–4 | Address fields (Welsh label) |
| 30 | cod_post | Postcode (Welsh label) |
| 31 | rhif_ffon | Phone number (Welsh label) |
| 32 | disgyblion | Pupils (Welsh label) |
| 33 | geom | WKT point geometry (Easting/Northing in British National Grid) |
 
</details>

---

## Limitations 

- Scotland is excluded because property transactions are managed by Registers of Scotland, a separate system from HM Land Registry, with a different data format and access process. Integrating both systems would be much more challenging.


---

## Any Extra Notes 