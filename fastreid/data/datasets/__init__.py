'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-22 14:44:35
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-22 15:11:53
FilePath: /IDA-nr/fastreid/data/datasets/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from ...utils.registry import Registry

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """
Registry for datasets
It must returns an instance of :class:`Backbone`.
"""

# Person re-id datasets
from .cuhk03 import CUHK03

from .dukemtmcreid import DukeMTMC, DukeMTMC_1, DukeMTMC_2, DukeMTMC_3

from .market1501 import Market1501
from .msmt17 import MSMT17

from .cuhk03_new_all import CUHK03All
from .market1501_all import Market1501All
from .msmt17_all import MSMT17All
from .cuhk02_all import CUHK02All

### small datasets
from .grid import GRID
from .iLIDS import iLIDS
from .prid import PRID
from .viper import VIPeR

from .AirportALERT import AirportALERT
# from .iLIDS import iLIDS
from .pku import PKU
from .prai import PRAI
from .saivt import SAIVT
from .sensereid import SenseReID
from .sysu_mm import SYSU_mm
from .thermalworld import Thermalworld
from .pes3d import PeS3D
from .caviara import CAVIARa
# from .viper import VIPeR
from .lpw import LPW
from .shinpuhkan import Shinpuhkan
from .wildtracker import WildTrackCrop
from .cuhk_sysu import cuhkSYSU

# Vehicle re-id datasets
from .veri import VeRi
from .vehicleid import VehicleID, SmallVehicleID, MediumVehicleID, LargeVehicleID
from .veriwild import VeRiWild, SmallVeRiWild, MediumVeRiWild, LargeVeRiWild

from .randperson import RandPerson, RandPerson_1, RandPerson_2 , RandPerson_3


__all__ = [k for k in globals().keys() if "builtin" not in k and not k.startswith("_")]
