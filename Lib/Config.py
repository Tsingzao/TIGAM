from easydict import EasyDict as edict

config = edict()

config.train = edict()


config.test = edict()


config.valid = edict()


config.evaluate = edict()


config.data = edict()
config.data.flipH = True
config.data.flipV = True
config.data.rotate = True
config.data.ratio = 0.1
config.data.seed = 20190906
config.data.areaList = ['','NEC','NCN','CCN','SCN','NWC','SWC','XJ','XZ']
config.data.CMPAS_Path = '/mnt/data133/data/CMPAS3H/'
config.data.CMPAS_yearList = ['2017', '2018']
config.data.CMPAS_monthList = [6,7,8,9]
config.data.CMPAS_format = 'MSP1_PMSC_AIWSRPF_CMPAS-3H-0p01_L88_%s_%s%02d%02d%02d00_00000-00000.nc'
config.data.NMC_Path = '/mnt/data169/data/AIWSRPF/NMCORI/'
config.data.NMC_yearList = ['2018']
config.data.NMC_monthList = [6,7,8,9]
config.data.NMC_format = 'MSP2_PMSC_AIWSRPF_NMCORI_L88_%s_%s%s00_00000-07200.nc'
config.data.range = {"NEC":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}},
                     "NCN":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}},
                     "CCN":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}},
                     "SCN":{"lon":{"min":104.0,"max":124.0}, "lat":{"min":15.0,"max":27.0}},
                     "NWC":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}},
                     "SWC":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}},
                     "XJ":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}},
                     "XZ":{"lon":{"min":0,"max":0}, "lat":{"min":0,"max":0}}}
config.data.size = {"NEC":{"h":0,"w":0},"NCN":{"h":0,"w":0},"CCN":{"h":0,"w":0},"SCN":{"h":1201,"w":2001},
                    "NWC":{"h":0,"w":0},"SWC":{"h":0,"w":0},"XJ":{"h":0,"w":0},"XZ":{"h":0,"w":0}}

