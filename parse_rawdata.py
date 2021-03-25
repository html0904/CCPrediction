import os
import numpy as np

os.chdir('.')
f=open('pdb_final.txt','r')

# list of categories aa_map sorted alphabetically, rest sorted by frequency
aa_map = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
all_solvents = ['VAPORDIFFUSION', 'HANGINGDROP', 'SODIUM', 'SITTINGDROP', 'ACETATE', 'TRIS', 'AMMONIUM', 'CHLORIDE', 'SULFATE', 'PEG3350', 'HEPES', 'CITRATE', 'MAGNESIUM', 'LITHIUM', 'PEG4000', 'HYDROCHLORIC', 'MES', 'PEG8000', 'BIS', 'PHOSPHATE', 'GLYCEROL', 'GLYCOL', 'POTASSIUM', 'IODIDE', 'PEG400', 'CACODYLATE', 'PEG6000', 'PEG', 'MPD', 'CALCIUM', 'DTT', 'FORMATE', 'SULPHATE', 'PROPANE', 'MANGANESE', 'PEG20000', 'ISOPROPANOL', 'MME', 'IMIDAZOLE', 'TARTRATE', 'NANODROP', 'ZINC', 'FLUORIDE', 'PEG2000', 'MALONATE', 'PEG1000', 'BROMIDE', 'BICINE', 'NITRATE', 'PEG1500', 'CITRIC', 'HYDROXYL', 'ATP', 'AZIDE', 'PEG3000', '2PROPANOL', 'EDTA', 'ETHANOL', 'TCEP', 'ONOMETHYLETHER', 'THIOCYANATE', 'OIL', '4PENTANEDIOL', 'TACSIMATE', 'NAD', 'DIMETHYLSULFOXIDE', 'MOPS', 'CHES', 'DIBASIC', 'CADMIUM', 'AMP', 'DIOXANE', 'GLYCINE', 'DEG', 'COBALT', 'PEG200', 'PEG300', 'PEG10000', 'CYANATE', 'JEFFAMINE', 'LIPIDIC', 'HEXANEDIOL', 'PEG550', 'PEG600', 'MALIC', 'TETRAHYDRATE', 'CAPS', 'PEG5000', 'MORPHEUS', 'CESIUM', 'PENTAERYTHRITOL', 'BME', 'ADA', 'SUCCINIC', 'DITHIOTHREITOL', 'PIPES', 'ADP', 'SUCROSE', 'BETAMERCAPTOETHANOL', 'MERCAPTOETHANOL', 'GLUCOSIDE', 'GLUCOSE', 'AMSO4', 'MCSG1', 'MALATE', 'PACT', 'BTP', 'NOXIDE', 'MPEG', 'MMT', 'PDTP', 'ETHOXYLATE', 'COA', 'SUCCINATE', 'TRIMETHYLAMINE', 'NICKEL', 'BES', 'METHANOL', '4DIOXANE', 'SPERMINE', '4BUTANEDIOL', 'SPG', 'BENZAMIDINE', 'PROPOXYLATE', 'UREA', 'AMPPNP', 'ETHER', 'MIB', 'PLP', 'METHANE', 'COPPER', 'PROPANOL', 'MORPHOLINOETHANESULFONIC', 'XYLITOL', 'BUTANOL', 'ALANINE', 'HYDROCHLORIDE', 'TRIZMA', 'SPERMIDINE', '3PROPANEDIOL', 'PENTANEDIOL', 'DITHIONITE', 'GDP', 'LPROLINE', 'TERTBUTANOL', 'GUANIDINE', 'LDAO', '1BUTANOL', 'POLYACRYLIC', 'POLYPROPYLENE', 'GLUTATHIONE', 'CACODYLIC', 'ACETONE', 'OXAMATE', 'ADENOSINE', 'CHAPS', 'FAD', 'PROLINE', 'GSH', 'PCTP', 'BERYLLIUM', 'BUTANEDIOL', 'CARBOXYLIC', 'DDT', 'TASCIMATE', 'HEPPS', 'TREHALOSE', 'HYDROXIDE', 'TRYPTONE', 'TAPS', 'BOG', 'TMAO', 'STRONTIUM', 'CAPSO', 'HEXAMINE', 'BORATE', 'ETHYLENEGLYCAL', 'RUBIDIUM', 'P3350', 'CARBONIC', 'MERCURY', 'BENZOATE', 'SDS', 'ACETONITRILE', 'PEGMME', 'METAHNOL', 'PROPANEDIOL', 'SULFITE', '2MERCAPTOETHANOL', 'PEG750', '6HEXANEDIOL', 'DLMALIC']
all_temps = ['293', '298', 'NULL', '277', '291', '295', '289', '294', '290', '292', '293.15', '296', '297', '277.15', '300', '291.15', '288', '278', '283', '287', '285', '273', '281', '298.15', '279', '289.15', '286', '280', '295.15', '303', '294.15', '310', '282', '292.15', '299', '295.5', '301', '100', '296.15', '287.15', '276', '284', '290.15', '293.5', '291.5', '281.15', '283.15', '293.2', '288.15', '291.2', '303.15', '302', '275', '297.15', '323', '274', '293.1', '277.2', '110', '285.15', '291.16', '277.5', '282.15', '279.15', '293.4', '316', '277.16', '281.16', '308', '289.1', '291.1', '277.13', '290.9', '309', '318', '294.5', '286.15', '311', '277.1', '315', '300.15', '310.15', '282.1', '273.15', '305', '295.7', '312', '296.4', '313', '293.14', '292.16', '279.5', '293.16', '296.2', '314', '278.15', '293.17', '293.65', '295.4', '277.4', '295.16', '295.65', '291.4', '295.1', '277.14', '298.5', '295.6', '277.45', '294.2', '304', '302.15', '289.16', '301.15', '277.12', '287.1', '293.13', '294.16', '319', '293.25', '289.2', '298.1', '296.35', '295.2', '297.3', '321', '255', '268', '292.2', '294.1', '298.13', '289.5', '288.2', '281.65', '292.3', '291.14', '276.15', '292.5', '297.5', '318.15', '290.05', '270', '105', '283.18', '106', '289.3', '271', '296.5', '291.3', '294.65', '280.65', '283.4', '292.8']

# SEQUENCE, PROCESSED_CONDITION, TEMPERATURE,PH, SOLVENT_CONTENT, MATTHEWS_COEFFICIENT
title = f.readline().strip('\n').split('\t')
line = f.readline()
X = []
Y = []
seq_len = []
misc_info = []
max_len = 2000 #range 2-8184
total = 71659
X = np.zeros([total, max_len*len(aa_map)],dtype=bool)
Y = np.zeros([total, len(all_solvents)], dtype=bool)
i=0

while line:
    elements = [text.strip(',') for text in line.strip('\n').split('\t')]

    # for aa sequence parsing
    seq = elements[0].split(',')
    aa_length = len(seq)
    if aa_length <= max_len:
       
        for aa in range(len(seq)):
            if seq[aa] in aa_map:
                X[i][len(aa_map)*aa+aa_map.index(seq[aa])] = 1
            
        # for solvents parsing
        solvents = elements[1].split(',');
        for s in solvents:
            Y[i][all_solvents.index(s)] = 1
        
        temperature = elements[2]
        pH = elements[3]
        solvent_content = elements[4]
        matthews_coefficient = elements[5]
        
        seq_len.append(aa_length)
        misc_info.append([temperature, pH, solvent_content, matthews_coefficient])
        i+=1
    
    line=f.readline()


seq_len = np.array(seq_len, dtype="int")
sorted_order = seq_len.argsort()
seq_len = seq_len[sorted_order]
X = X[sorted_order]
Y = Y[sorted_order]
misc_info = np.array(misc_info)
misc_info = misc_info[sorted_order]

np.save('aa_seq_onehot.npy', X) # one hot encoding of protein seq, flattened
np.save('solvents_onehot.npy', Y) # one hot encoding of solvents, see order above
np.save('seq_len.npy', seq_len) # seq len of each row in X
np.save('misc_info.npy', misc_info) # misc info of each row in X (temp, pH, solvent content, matthews coefficient)

    
