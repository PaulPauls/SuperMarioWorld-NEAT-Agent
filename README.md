## Reinforcement-Learning Agent trained with the NEAT algorithm to play Super Mario World ##

**AS OF MAY 2019: CODE AND README CURRENTLY UNDER HEAVY DEVELOPMENT**

Reinforcement Learning Agent trained in OpenAI's Retro environment with the game 'Super Mario World' for the SNES. This program makes heavy use of [NEAT-Python](https://github.com/codereclaimers/neat-python) and its standard configurations.

The in-game data OpenAI's retro environment provides for Super Mario World has been enhanced in the custom `data.json` provided, which should replace the standard `data.json` of super mario world. The ram addresses of this in-game data has been found through the [ROM RAM map of Super mario World](https://www.smwcentral.net/?p=nmap&m=smwram). The following in-game information has been added:
  * x_pos_player
  * midway_point_flag
  * timer_hundreds
  * timer_tens
  * timer_ones
  * level_exit_info



---

#### Setup and Execute ####

1. Replace the standard `data.json` in the installation directory of the retro package. Assuming Python 3.7 in a virtualenv:
```
$ cp -f ./data.json ./venv/lib/python3.7/site-packages/retro/data/stable/SuperMarioWorld-Snes/
```

2. Configure HyperParameters of `neat-config` as well as those in `evolve-agent.py`

3. Execute `$ python3 evolve-agent.py`

