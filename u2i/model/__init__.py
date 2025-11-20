
from .basemodel import BaseModel
from .sasrec import SASRec
from .bert4rec_lite import Bert4RecLite
from .hstu_lite import HSTURec
from .sasrec_addfeat import SASRecAddFeat

MODELS = {
    "basemodel": BaseModel,
    "sasrec": SASRec,
    "bert4rec": Bert4RecLite,
    "hstu": HSTURec,
    "sasrec_addfeat": SASRecAddFeat
}