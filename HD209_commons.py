import iris
from pathlib import Path

SUITES = {"hd209458b": {
            "equilibrium": {
                "solar": {
                    "planet": "hd209458b",
                    "suite": "u-bk852",
                    "dir_for_merged": Path.home().parent.parent/"data"/"mz355"/ "um_runs"/ "hd209458b"/ "equilibrium"/ "u-bk852"/ "merged",   
                },
            },
            "kinetics": {
                "solar": {
                    "planet": "hd209458b",
                    "suite": "u-bk871",
                    "dir_for_merged": Path.home().parent.parent/"data"/"mz355"/ "um_runs"/ "hd209458b"/ "kinetics"/ "u-bk871"/ "merged" 
                },
            },
        },
}