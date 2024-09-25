import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk
import seaborn as sns
from functions import show_frame, visualize_barplot, visualize_scatter,count_words
import matplotlib.pyplot as plt
plt.ion()

data = pd.read_csv('data_first_preprocessing.csv',index_col=0)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test,c_train,c_test = train_test_split(data.drop(["outliers",'price'],axis=1),data['price'],data['outliers'],test_size = 0.2,random_state=42)


#at first we will working with columns with type object
object_columns = data.select_dtypes(include='object').columns.tolist()
print(object_columns)
# 'brand', 'model', 'fuel_type', 'ext_col', 'int_col'

# get extra information from fuel_type and brand
from category_encoders import MEstimateEncoder
encoder = MEstimateEncoder(cols=["fuel_type","brand"], m=1.0)
X_train = encoder.fit_transform(X_train,y_train)
X_test = encoder.transform(X_test)

#let's figure out what words are most popular in columns 'ext_col', 'int_col','model'. Which characterised exspensive cars?

# print(count_words(X_train,'model'))
most_common_words_model = [('Base', 23896), ('Rover', 13641), ('Premium', 10820), ('S', 9219), ('Sport', 8533), ('Range', 7668), ('4MATIC', 7278), ('i', 6427), ('1500', 6365), ('XLT', 5773), ('AMG', 5492), ('350', 5415), ('F-150', 4897), ('Limited', 4795), ('Plus', 4759), ('911', 4222), ('xDrive', 4114), ('SE', 3832), ('GT', 3724), ('2.0T', 3674), ('HSE', 3260), ('Corvette', 3213), ('Carrera', 3111), ('Mustang', 3022), ('Model', 3006), ('C', 2875), ('300', 2863), ('Camaro', 2851), ('63', 2838), ('E', 2801), ('Platinum', 2792), ('E-Class', 2618), ('Wrangler', 2328), ('M3', 2252), ('Suburban', 2238), ('Escalade', 2165), ('Supercharged', 2157), ('Expedition', 2153), ('F-250', 2140), ('Lariat', 2064), ('3.0T', 2043), ('Luxury', 1982), ('LT', 1947), ('Stingray', 1940), ('RX', 1892), ('GX', 1887), ('M4', 1777), ('550', 1771), ('460', 1756), ('Touring', 1741), ('Grand', 1725), ('Coupe', 1716), ('2500', 1708), ('Edition', 1695), ('Silverado', 1673), ('G', 1634), ('Prestige', 1606), ('Turbo', 1542), ('GLE', 1508), ('C-Class', 1500), ('Long', 1487), ('LTZ', 1472), ('Cab', 1400), ('Laramie', 1379), ('ESV', 1371), ('SL', 1369), ('2SS', 1366), ('Cayenne', 1365), ('Explorer', 1337), ('SR5', 1269), ('SLT', 1240), ('L', 1232), ('LX', 1228), ('Package', 1217), ('S-Class', 1205), ('Highlander', 1158), ('X', 1157), ('Performance', 1150), ('SL-Class', 1130), ('Gladiator', 1120), ('V6', 1114), ('Q5', 1090), ('Hybrid', 1070), ('Tundra', 1069), ('RS', 1065), ('Competition', 1050), ('Premier', 1045), ('Y', 1035), ('43', 1033), ('450', 1028), ('Bronco', 1020), ('GLC', 1005), ('F', 997), ('Sierra', 995), ('Tahoe', 994), ('Cherokee', 989), ('Macan', 983), ('Reserve', 979), ('Big', 947), ('55', 922), ('LS', 918), ('335', 914), ('XLE', 896), ('A6', 875), ('IS', 869), ('Cooper', 867), ('XL', 865), ('QX60', 863), ('w/2LT', 852), ('4Runner', 839), ('Gran', 827), ('328', 823), ('Camry', 817), ('250', 816), ('Max', 814), ('4.0T', 810), ('A4', 801), ('Unlimited', 799), ('EX', 794), ('4S', 782), ('3.0L', 763), ('Yukon', 762), ('Cruiser', 759), ('Panamera', 757), ('Denali', 754), ('7', 742), ('4-Door', 737), ('Advanced', 737), ('GLS', 735), ('Roadster', 735), ('SV', 726), ('Ram', 720), ('R-Dynamic', 720), ('Horn', 717), ('SLE', 713), ('Discovery', 678), ('A5', 675), ('Evoque', 668), ('WRX', 664), ('S4', 663), ('CTS', 658), ('R', 653), ('Tacoma', 652), ('1SS', 644), ('Sahara', 639), ('Sequoia', 636), ('line', 634), ('M5', 619), ('Golf', 617), ('Rubicon', 616), ('quattro', 614), ('ES', 609), ('Challenger', 608), ('G-Class', 606), ('Navigator', 603), ('Dynamic', 598), ('w/3LT', 593), ('Transit-350', 591), ('Genesis', 588), ('High', 588), ('330', 585), ('Impreza', 583), ('Velar', 582), ('Overland', 581), ('LWB', 575), ('53', 574), ('GL', 564), ('3.8', 561), ('MKZ', 550), ('ST', 549), ('Defender', 548), ('SX', 548), ('Romeo', 544), ('Telluride', 535), ('TRD', 533), ('SEL', 532), ('3', 530), ('Continental', 522), ('Acadia', 508), ('Civic', 504), ('Roof', 501), ('A7', 499), ('MDX', 498), ('SWB', 496), ('M6', 493), ('GL-Class', 492), ('Fusion', 489), ('King', 485), ('Ranch', 485), ('1LT', 485), ('GTS', 481), ('Q7', 481), ('Land', 477), ('Sorento', 477), ('R1S', 476), ('470', 466), ('Cayman', 465), ('Outback', 463), ('435', 463), ('560', 463), ('QX80', 462), ('TLX', 457), ('R/T', 455), ('Liberty', 455), ('CX-30', 450), ('Boxster', 448), ('A3', 448), ('w/1LT', 447), ('750', 443), ('Signature', 441), ('Crew', 440), ('V8', 435), ('SuperCab', 433), ('EX-L', 433), ('Avalanche', 432), ('Metris', 429), ('Solara', 426), ('650', 425), ('Cabriolet', 424), ('Raptor', 417), ('Mazda6', 414), ('GT3', 409), ('F-TYPE', 405), ('3500', 401), ('Focus', 398), ('Super', 398), ('Duty', 398), ('Q4', 391), ('SRT', 387), ('Sprinter', 386), ('Ti', 381), ('570', 380), ('S5', 380), ('Adventure', 378), ('Country', 375), ('Giulia', 373), ('400', 370), ('Outer', 368), ('Banks', 368), ('Ghibli', 366), ('LE', 365), ('GTI', 364), ('Durango', 363), ('Bentayga', 363), ('P530', 359), ('Trail', 358), ('Q8', 358), ('4', 357), ('Select', 357), ('SL500', 354), ('Pathfinder', 351), ('Three-Row', 345), ('Enclave', 345), ('F-350', 344), ('CX-9', 343), ('M8', 342), ('CS', 340), ('Optima', 337), ('MX-5', 333), ('Miata', 333), ('A8', 332), ('Armada', 331), ('Edge', 331), ('Quad', 329), ('EL', 329), ('Charger', 328), ('ALPINA', 325), ('P250', 319), ('Avalon', 319), ('Ultimate', 319), ('Mazda3', 318), ('528', 318), ('2.5i', 317), ('500', 317), ('2.5', 317), ('2.0L', 316), ('B7', 314), ('Pilot', 313), ('Standard', 311), ('Seat', 309), ('RST', 308), ('SLT-1', 306), ('Town', 305), ('R8', 302), ('Boss', 300), ('Technology', 299), ('AWD', 298), ('SPORT', 298), ('X3', 297), ('Excursion', 297), ('3.0', 295), ('EcoBoost', 295), ('428', 294), ('CLA', 293), ('Advance', 291), ('Shelby', 288), ('X5', 287), ('Sportage', 287), ('Red', 286), ('Traverse', 283), ('FJ', 282), ('Dakota', 281), ('Laredo', 281), ('Rebel', 281), ('Jetta', 279), ('G37', 273), ('GV70', 271), ('CLA-Class', 270), ('350Z', 270), ('430', 269), ('Santa', 265), ('Huracan', 265), ('5.2', 264), ('Pro', 260), ('Transit', 259), ('Connect', 259), ('Tech', 258), ('5.0L', 257), ('75D', 256), ('Mach-E', 254), ('&', 253), ('2.0i', 252), ('Colorado', 248), ('LC', 248), ('Q50', 247), ('2LT', 247), ('450h', 247), ('Fe', 246), ('Z06', 246), ('GT-R', 244), ('Handling', 244), ('xDrive30i', 244), ('Titan', 242), ('Sonata', 242), ('Type', 240), ('Express', 240), ('SQ5', 239), ('3.0t', 238), ('ILX', 236), ('Frontier', 234), ('S-Model', 234), ('Essence', 234), ('SLK-Class', 233), ('Pure', 232), ('P525', 231), ('Hellcat', 230), ('Spyder', 230), ('Scat', 230), ('Pack', 230), ('H2', 230), ('F-PACE', 228), ('Timberline', 228), ('718', 227), ('Xterra', 227), ('TT', 226), ('Track', 225), ('Westminster', 225), ('Tradesman', 225), ('3.6L', 224), ('Ghost', 224), ('540', 221), ('Luxe', 220), ('Envision', 220), ('45', 218), ('2.5T', 217), ('SVR', 217), ('S2000', 217), ('BRZ', 217), ('LUXURY', 216), ('Terrain', 216), ('Z71', 213), ('x', 213), ('V10', 211), ('740', 211), ('PreRunner', 210), ('ATS', 207), ('Autobiography', 207), ('535', 207), ('ProMaster', 206), ('HST', 206), ('Tucson', 205), ('T6', 204), ('P400', 204), ('Escape', 204), ('RC', 201), ('ZR2', 200), ('Van', 200), ('Bend', 199), ('Eclipse', 199), ('392', 199), ('Classic', 198), ('Line', 198), ('Carbon', 197), ('3.5T', 194), ('MHEV', 193), ('CLK-Class', 193), ('Solstice', 188), ('M', 188), ('SULEV', 188), ('Extended', 187), ('M240', 186), ('SS', 185), ('Firebird', 182), ('Badlands', 181), ('P100D', 180), ('Martin', 179), ('GT500', 178), ('GLA', 177), ('Taycan', 176), ('Murano', 175), ('Car', 174), ('T5', 174), ('QX56', 174), ('Urus', 173), ('G70', 173), ('Spider', 173), ('Cullinan', 172), ('H3', 172), ('Stelvio', 171), ('3.3T', 170), ('RDX', 170), ('325', 170), ('LR4', 169), ('Rogue', 169), ('Compass', 168), ('X6', 168), ('TL', 167), ('W12', 167), ('Renegade', 167), ('XC90', 166), ('100D', 166), ('GS', 166), ('XF', 165), ('MKC', 165), ('GXP', 165), ('3.2', 165), ('G35', 164), ('Cruze', 164), ('X-Dynamic', 163), ('M2', 162), ('CLK', 161), ('xDrive40i', 160), ('Accord', 158), ('Special', 157), ('GranSport', 157), ('Double', 153), ('Maxima', 151), ('Work', 151), ('580', 151), ('RAV4', 151), ('CR-V', 150), ('CX-5', 148), ('Mega', 148), ('Launch', 147), ('3.7L', 147), ('370Z', 146), ('Countryman', 146), ('Leather', 146), ('Wildtrak', 146), ('Sti', 145), ('XTS', 145), ('Access', 143), ('Speed', 143), ('ALL4', 143), ('w/Technology', 142), ('Inscription', 142), ('Monte', 140), ('Carlo', 140), ('570S', 140), ('Altima', 140), ('Ranger', 139), ('S6', 139), ('TSI', 139), ('Lightning', 138), ('Wagoneer', 136), ('Stinger', 136), ('Q3', 136), ('Corolla', 135), ('Series', 135), ('M550', 134), ('SL550', 134), ('Momentum', 133), ('2', 133), ('2.5L', 132), ('Juke', 132), ('Quattroporte', 131), ('X1', 131), ('EVO', 130), ('Malibu', 129), ('G80', 129), ('5', 129), ('Lancer', 129), ('Titanium', 128), ('4WD', 128), ('S3', 127), ('XT5', 127), ('First', 126), ('M440', 124), ('Q60', 124), ('Z4', 124), ('NX', 123), ('S60', 122), ('Z28', 122), ('E350', 120), ('4.2', 118), ('LUX', 117), ('Landmark', 117), ('340', 117), ('Enthusiast', 117), ('Trailblazer', 117), ('Sky', 116), ('Thunderbird', 115), ('Ascent', 115), ('7-Passenger', 115), ('Plug-In', 115), ('GR86', 114), ('III', 114), ('G90', 113), ('ZL1', 113), ('STI', 113), ('X7', 113), ('Element', 113), ('1.4T', 113), ('AT4', 112), ('530', 110), ('Warlock', 109), ('135', 109), ('GT2', 109), ('GranTurismo', 109), ('Palisade', 108), ('xDrive28i', 108), ('SportWagen', 107), ('S8', 106), ('CLS-Class', 106), ('IS-F', 105), ('Black', 105), ('Utility', 105), ('Police', 105), ('Interceptor', 105), ('w/Advance', 105), ('Supra', 104), ('Aventador', 104), ('w/Performance', 103), ('CLS', 103), ('Modena', 103), ('Forester', 102), ('M235', 102), ('XT', 102), ('I', 102), ('Taurus', 101), ('Crafted', 100), ('SRT8', 100), ('Trans', 100), ('Am', 100), ('Vantage', 100), ('R-Line', 100), ('SLK', 100), ('C2S', 99), ('Preferred', 98), ('Deluxe', 98), ('PMC', 98), ('GT-Line', 96), ('TSX', 96), ('Avenir', 94), ('MKX', 91), ('600', 91), ('Tire', 90), ('P380', 90), ('PHEV', 90), ('GT4', 90), ('TDI', 88), ('Tradesman/Express', 88), ('SR', 88), ('Ridgeline', 87), ('488', 87), ('2.9T', 87), ('LARIAT', 87), ('Touring-L', 87), ('H/D', 85), ('Custom', 85), ('Gallardo', 85), ('25t', 85), ('CC', 85), ('Quadrifoglio', 84), ('Outlander', 84), ('plus', 84), ('w/A-Spec', 83), ('C300', 83), ('Nautilus', 83), ('SENSORY', 82), ('Legacy', 82), ('xDrive45e', 82), ('SXL', 82), ('GTC', 81), ('GT350', 81), ('3.5L', 80), ('XLS', 80), ('1.8', 79), ('Ci', 79), ('e-tron', 79), ('35t', 77), ('M-Class', 77), ('ML', 77), ('XD', 77), ('SL63', 77), ('Passenger', 77), ('sDrive28i', 77), ('5.0', 77), ('SRX', 77), ('Autobahn', 76), ('w/2LZ', 76), ('Forte', 76), ('RSX', 76), ('w/Premium', 75), ('Versa', 75), ('M760', 75), ('Longhorn', 74), ('FWD', 74), ('A-Class', 73), ('A', 73), ('220', 73), ('GV80', 73), ('Odyssey', 73), ('LaCrosse', 73), ('Altitude', 73), ('SHO', 72), ('SC', 72), ('T8', 72), ('QX70', 72), ('Impala', 72), ('LR2', 72), ('Evolution', 72), ('T', 71), ('CX-7', 71), ('Levante', 71), ('Diesel', 70), ('Atlas', 70), ('640', 70), ('Onyx', 70), ('SLE-2', 70), ('Crosstrek', 70), ('SVAutobiography', 69), ('CTS-V', 69), ('XK8', 69), ('VR6', 69), ('Hardtop', 69), ('SVT', 68), ('XC60', 68), ('SL600', 68), ('DTS', 67), ('Off', 67), ('Road', 67), ('Elite', 66), ('s', 66), ('A-Spec', 66), ('E-Hybrid', 66), ('w/DCC', 66), ('Navigation', 66), ('300C', 65), ('228', 65), ('2.4', 65), ('Anniversary', 65), ('Azure', 65), ('Superleggera', 64), ('RTL-E', 64), ('K5', 63), ('Accent', 63), ('SLK320', 63), ('Flex', 62), ('M340', 62), ('Blazer', 62), ('1.8T', 62), ('4x4', 62), ('GranLusso', 61), ('CT', 61), ('200h', 61), ('Beetle', 61), ('Air', 61), ('3.6R', 61), ('Pro-4X', 61), ('Nightfall', 61), ('xi', 61), ('tC', 60), ('Equinox', 60), ('MKS', 60), ('Equus', 60), ('Cargo', 59), ('Roma', 59), ('C70', 58), ('SVJ', 58), ('is', 58), ('i3', 58), ('SuperCrew', 58), ('Squared', 58), ('GLA-Class', 57), ('iPerformance', 57), ('WT', 57), ('performance', 57), ('SLC', 57), ('Tremor', 57), ('Mojave', 56), ('S7', 56), ('Quest', 56), ('DRW', 56), ('EXT', 56), ('Recharge', 56), ('Dart', 56), ('GTC4Lusso', 55), ('Aviator', 55), ('330e', 55), ('Freedom', 55), ('Collection', 55), ('R-Sport', 55), ('Plaid', 55), ('CR', 54), ('i8', 54), ('LT1', 54), ('Elantra', 53), ('xDrive35i', 53), ('840', 53), ('II', 53), ('Corsair', 53), ('GLX', 53), ('RX-8', 53), ('4xe', 53), ('Q70', 53), ('440', 52), ('R-Spec', 52), ('NISMO', 52), ('Canyon', 52), ('X-Line', 52), ('720S', 52), ('XT6', 51), ('Elevation', 51), ('GLK-Class', 51), ('GLK', 51), ('Arteon', 51), ('GT1', 50), ('Maybach', 50), ('Trailhawk', 50), ('Si', 50), ('LP580-2S', 50), ('xDrive50i', 49), ('Veloster', 49), ('Pacifica', 49), ('EV', 49), ('GLI', 48), ('S80', 48), ('Hard', 48), ('Rock', 48), ('Cobra', 47), ('Appearance', 47), ('NV', 47), ('NV3500', 47), ('HD', 47), ('Matrix', 47), ('Commander', 47), ('CT5-V', 47), ('iSport', 46), ('VE', 46), ('AUTOGRAPH', 46), ('200t', 46), ('TTS', 46), ('XJ8', 46), ('SSR', 45), ('RL', 45), ('M40i', 45), ('300h', 45), ('Summit', 45), ('Lux', 44), ('XE', 44), ('Marauder', 44), ('Leaf', 44), ('TRX', 44), ('Passat', 44), ('200', 43), ('Sienna', 43), ('DBS', 43), ('C-HR', 43), ('Grecale', 43), ('EQS', 43), ('3.7X', 43), ('Vanden', 43), ('Plas', 43), ('SRT-10', 42), ('Energi', 42), ('w/Range', 42), ('Extender', 42), ('Crosstour', 42), ('Sportback', 42), ('SXT', 42), ('4.6', 41), ('Crossfire', 41), ('Bolt', 41), ('Touareg', 41), ('Phantom', 41), ('Integra', 41), ('Shinsen', 40), ('S500', 40), ('Heritage', 40), ('Prius', 40), ('XSE', 39), ('Clubman', 39), ('Widebody', 38), ('300M', 38), ('A91', 38), ('Li', 38), ('GTB', 38), ('w/Preferred', 38), ('Magnum', 38), ('Wagon', 38), ('Drophead', 38), ('Passport', 37), ('LSE', 37), ('Entertainment', 37), ('Pkgs', 37), ('Transit-250', 37), ('E500', 37), ('Niro', 37), ('Premiere', 37), ('Willys', 37), ('XK', 37), ('Seltos', 36), ('DBX', 36), ('Cross', 36), ('QX30', 36), ('LR3', 36), ('RF', 36), ('XJ', 36), ('750i', 35), ('LP580-2', 35), ('Window', 35), ('Eddie', 35), ('Bauer', 35), ('S-10', 35), ('SLK230', 35), ('Kompressor', 35), ('3.7', 35), ('SLK280', 35), ('110', 35), ('M50i', 34), ('Trac', 34), ('Encore', 34), ('EUV', 33), ('330i', 33), ('Label', 33), ('Celica', 33), ('CXL', 33), ('MC', 33), ('1LZ', 32), ('Murcielago', 32), ('Journey', 32), ('Kona', 32), ('EV6', 32), ('302', 32), ('Auto', 32), ('Horn/Lone', 31), ('Star', 31), ('450+', 31), ('SLT-2', 31), ('2.4L', 31), ('DeVille', 31), ('Q70h', 31), ('Maverick', 30), ('Arnage', 30), ('X4', 30), ('GT350R', 29), ('SQ7', 29), ('Calligraphy', 29), ('Eurovan', 29), ('MV', 29), ('Targa', 29), ('N', 28), ('XR', 28), ('Blackwing', 28), ('85D', 28), ('FX37', 28), ('MR', 28), ('30t', 28), ('FR-S', 28), ('Monogram', 28), ('Latitude', 28), ('C-Max', 27), ('Z51', 27), ('1', 27), ('California', 27), ('XC70', 26), ('iL', 26), ('E-PACE', 26), ('Viper', 26), ('Savana', 26), ('V-Series', 26), ('CX', 26), ('MC20', 26), ('Q40', 25), ('MazdaSpeed3', 25), ('430i', 25), ('LP610-4', 25), ('86', 25), ('50', 25), ('F12berlinetta', 25), ('Tecnica', 25), ('Evora', 24), ('C4S', 24), ('Avenger', 24), ('28i', 24), ('New', 24), ('FX4', 24), ('Activity', 24), ('British', 24), ('Design', 24), ('1794', 24), ('Wraith', 24), ('DE', 24), ('Cheyenne', 23), ('Venture', 23), ('Silver', 23), ('Caravan', 23), ('Capstone', 23), ('WK', 23), ('Nightshade', 23), ('FX50', 23), ('Routan', 23), ('SLE-1', 23), ('R-Design', 23), ('Club', 23), ('Koup', 23), ('3500XD', 23), ('300ZX', 23), ('M760i', 23), ('NSX', 23), ('Flying', 22), ('Spur', 22), ('LXS', 22), ('440i', 22), ('S40', 22), ('Cascada', 22), ('65', 22), ('Prelude', 22), ('SH', 22), ('Alltrack', 22), ('LP570-4', 21), ('Turismo', 21), ('w/Rear', 21), ('Symmetrical', 21), ('Doors', 21), ('Flareside', 21), ('Montero', 21), ('C280', 21), ('SLE1', 21), ('Kicks', 20), ('M850', 20), ('CT6', 20), ('NV200', 20), ('323', 20), ('LP550-2', 20), ('Cruz', 19), ('Powerwagon', 19), ('MKT', 19), ('Protege', 19), ('DX', 19), ('sport', 19), ('Normal', 19), ('1XL', 19), ('3.0i', 19), ('XRS', 19), ('A-SPEC', 19), ('Packages', 19), ('2.0', 18), ('Rainier', 18), ('Two', 18), ('SLS', 18), ('S90', 18), ('E250', 18), ('120Ah', 17), ('Release', 17), ('6.0', 17), ('Sonic', 17), ('FF', 17), ('Caprice', 17), ('PLUS', 17), ('Eos', 17), ('Demon', 17), ('C55', 17), ('X-Pro', 17), ('9-3', 17), ('F430', 17), ('Berlinetta', 17), ('Baja', 17), ('Z85', 17), ('85', 17), ('3.5', 16), ('90D', 16), ('Tiguan', 16), ('GS-R', 16), ('Automatic', 16), ('SQ8', 16), ('M56', 16), ('128', 16), ('spec.B', 16), ('20th', 16), ('Grade', 16), ('LP700-4', 16), ('G8', 16), ('XJ6', 16), ('P90D', 16), ('R3', 16), ('A91-MT', 15), ('C30', 15), ('X2', 15), ('B5', 15), ('Sebring', 15), ('LP750-4', 15), ('Superveloce', 15), ('G6', 15), ('GTP', 15), ('E55', 15), ('Portfolio', 15), ('Lucerne', 15), ('Twin', 15), ('GSR', 15), ('Convenience', 14), ('Bullitt', 14), ('Elise', 14), ('CT4', 14), ('80th', 14), ('3.0si', 14), ('CE', 14), ('IX', 14), ('CT5', 14), ('GTO', 14), ('2LZ', 14), ('TrailSport', 14), ('ZDX', 14), ('525', 14), ('XFR-S', 13), ('LP560-4', 13), ('W/T', 13), ('V60', 13), ('Venza', 13), ('Pearl', 13), ('Capsule', 13), ('40', 13), ('250C', 13), ('v', 13), ('Three', 13), ('Mirai', 13), ('CX-50', 13), ('Fiesta', 12), ('Cube', 12), ('70D', 12), ('R1T', 12), ('Regal', 12), ('-', 12), ('230', 12), ('94', 12), ('Ah', 12), ('Revero', 12), ('DB7', 12), ('Volante', 12), ('M37', 12), ('RLX', 11), ('Caliber', 11), ('CLS500', 11), ('Final', 11), ('Convertible', 11), ('XB7', 11), ('Mark', 11), ('iM', 11), ('Route', 11), ('XC40', 11), ('CX-90', 11), ('ID.4', 11), ('sDrive35is', 10), ('60', 10), ('1.8L', 10), ('860', 10), ('Executive', 10), ('400E', 10), ('MazdaSpeed', 10), ('Electric', 10), ('GLB', 10), ('Yaris', 10), ('XKR', 10), ('4500', 9), ('Roadmaster', 9), ('Estate', 9), ('2+2', 9), ('sDrive35i', 9), ('Cambiocorsa', 9), ('HS', 9), ('250h', 9), ('Pickup', 9), ('Truck', 9), ('EcoSport', 9), ('SES', 9), ('xB', 9), ('Verano', 9), ('530e', 9), ('Trax', 9), ('IONIQ', 8), ('E150', 8), ('1.6', 8), ('20d', 8), ('Z', 8), ('Proto', 8), ('Spec', 8), ('C40', 8), ('SL400', 8), ('sDrive30i', 8), ('Lounge', 8), ('Patriot', 8), ('Prime', 7), ('320', 7), ('240SX', 7), ('VDC', 7), ('Aero', 7), ('350h', 7), ('D', 7), ('Wind', 7), ('CT6-V', 7), ('4.2L', 7), ('Fit', 7), ('Prowler', 6), ('LYRIQ', 6), ('SUT', 6), ('Vue', 6), ('4.0', 6), ('812', 6), ('Superfast', 6), ('Z3', 6), ('R-Class', 6), ('1HY', 6), ('SX4', 6), ('Veyron', 6), ('16.4', 6), ('GR', 6), ('Circuit', 6), ('Sentra', 5), ('Clarity', 5), ('25th', 5), ('Mid', 5), ('Capri', 5), ('XR2', 5), ('STS', 5), ('HEV', 5), ('Mirage', 5), ('124', 5), ('Abarth', 5), ('500X', 5), ('Trekking', 5), ('HUMMER', 4), ('Sedan', 4), ('850', 4), ('xDrive40e', 4), ('57', 4), ('bZ4X', 4), ('Plug-in', 4), ('John', 4), ('Works', 4), ('XT4', 4), ('e-Golf', 3), ('Insight', 3), ('allroad', 3), ('Rio', 3), ('eDrive', 3), ('Value', 3), ('K900', 3), ('ForTwo', 2), ('XLR', 2), ('ZR-1', 2), ('500e', 2), ('Battery', 2), ('740e', 2), ('V', 1), ('35i', 1)]

#witch words characterised most expensive cars?
#model
m=12
outliers = data[data['price'] > data['price'].quantile(0.90)]
#most_common_words_outliers = count_words(outliers,'model')
most_common_words_outliers = [('Base', 3155), ('911', 2096), ('S', 2053), ('AMG', 1734), ('Range', 1414), ('Carrera', 1256), ('4MATIC', 1103), ('63', 1005), ('Corvette', 924), ('GT', 889), ('Stingray', 793), ('Premium', 770), ('Sport', 768), ('G', 633), ('Turbo', 619), ('SE', 604), ('Competition', 578), ('Model', 556), ('xDrive', 531), ('M4', 474), ('1500', 438), ('7', 429), ('RS', 412), ('Escalade', 399), ('F-150', 382), ('Platinum', 376), ('Edition', 372), ('GLE', 368), ('Plus', 367)]
print(most_common_words_outliers)
mcw_outliers = [i[0] for i in most_common_words_outliers[:m]]
k =mcw_outliers
for fea in k:
    X_train[fea] = X_train['model'].map(lambda x: k.index(fea)+1 if fea.lower() in x.lower() else 0)
X_train['out_model'] = X_train[k].sum(axis=1)
X_train = X_train.drop(columns=k)

for fea in k:
    X_test[fea] = X_test['model'].map(lambda x: k.index(fea)+1 if fea.lower() in x.lower() else 0)
X_test['out_model'] = X_test[k].sum(axis=1)
X_test = X_test.drop(columns=k)
#let's check it out
visualize_barplot('out_model',pd.concat([X_train,y_train],axis =1))
#we need to drop from list word 'rover' because it generate too much noise
encoder = MEstimateEncoder(cols=['out_model'], m=5.0)
X_train = encoder.fit_transform(X_train,y_train)
X_test = encoder.transform(X_test)

X_test = X_test.drop('model',axis =1)
X_train = X_train.drop('model',axis =1)


#int_col
m=7
most_common_words_outliers = count_words(outliers,'int_col')
print(most_common_words_outliers)
mcw_outliers = [i[0] for i in most_common_words_outliers[:m]]
k =mcw_outliers
for fea in k:
    X_train[fea] = X_train['int_col'].map(lambda x: k.index(fea)+1 if fea.lower() in x.lower() else 0)
X_train['out_int_col'] = X_train[k].sum(axis=1)
X_train = X_train.drop(columns=k)

for fea in k:
    X_test[fea] = X_test['int_col'].map(lambda x: k.index(fea)+1 if fea.lower() in x.lower() else 0)
X_test['out_int_col'] = X_test[k].sum(axis=1)
X_test = X_test.drop(columns=k)



#let's check it out
visualize_barplot('out_int_col',pd.concat([X_train,y_train],axis =1))
encoder = MEstimateEncoder(cols=['out_int_col'], m=5.0)
X_train = encoder.fit_transform(X_train,y_train)
X_test = encoder.transform(X_test)

X_test = X_test.drop('int_col',axis =1)
X_train = X_train.drop('int_col',axis =1)

#ext_col
m=7
most_common_words_outliers = count_words(outliers,'ext_col')
print(most_common_words_outliers)
mcw_outliers = [i[0] for i in most_common_words_outliers[:m]]
k =mcw_outliers
for fea in k:
    X_train[fea] = X_train['ext_col'].map(lambda x: k.index(fea)+1 if fea.lower() in x.lower() else 0)
X_train['out_ext_col'] = X_train[k].sum(axis=1)
X_train = X_train.drop(columns=k)

for fea in k:
    X_test[fea] = X_test['ext_col'].map(lambda x: k.index(fea)+1 if fea.lower() in x.lower() else 0)
X_test['out_ext_col'] = X_test[k].sum(axis=1)
X_test = X_test.drop(columns=k)



#let's check it out
visualize_barplot('out_ext_col',pd.concat([X_train,y_train],axis =1))
encoder = MEstimateEncoder(cols=['out_ext_col'], m=5.0)
X_train = encoder.fit_transform(X_train,y_train)
X_test = encoder.transform(X_test)

X_test = X_test.drop('ext_col',axis =1)
X_train = X_train.drop('ext_col',axis =1)


show_frame(X_train)
(pd.concat([X_train,y_train,c_train],axis =1)).to_csv('train.csv')
(pd.concat([X_test,y_test,c_test],axis =1)).to_csv('test.csv')