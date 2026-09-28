# mecanum_ws

Isaac Lab workspace mecanum kerekes robot RL tanításához (manager-based, RSL-RL / skrl).

## Struktúra

```
mecanum_ws/
├── assets/robots/mecanum/          <- GENERÁLT robot: robot.urdf, meshes/roller.stl, wheel_geometry.json
├── scripts/
│   ├── rsl_rl/, skrl/              <- train / play
│   └── tools/
│       ├── generate_mecanum_robot.py   <- görgős mecanum robot URDF generátor
│       ├── validate_mecanum_wheels.py  <- nyílt hurkú fizikai validáció (nincs RL)
│       └── summarize_validation.py     <- validációs eredmények táblázata, PASS/FAIL
└── source/mecanum_ws/mecanum_ws/
    ├── robots/mecanum.py           <- robot paraméterek, DC motor, anyag (a generátorral szinkronban)
    └── tasks/manager_based/
        ├── mecanum_velocity/       <- sebességkövetés sík talajon
        └── mecanum_navigation/     <- célpontba navigálás oszlopok között
```

## Taskok

| Task | Action |
|---|---|
| `Mecanum-Velocity-Flat-v0` | a policy közvetlenül a 4 kerék sebességét adja ki |
| `Mecanum-Velocity-Flat-IK-v0` | a policy bázis twistet ad ki (vx, vy, wz), mecanum IK képezi le a kerekekre |
| `Mecanum-Navigation-Pillars-v0` | 4 kerék fordulat-alapjel (DC motoron át), end-to-end navigáció oszlopok között |
| `Mecanum-Navigation-Pillars-IK-v0` | ugyanez, bázis twisttel (IK) |

Mindegyiknek van `-Play-v0` változata. Minden task 360 Hz-es fizikával fut (decimation 7 → 51 Hz-es policy),
mert a görgős kerék validációja szerint ez alatt a kontaktus-zaj numerikus (lásd lent).

### Navigáció oszlopok között (`Mecanum-Navigation-Pillars-v0`)

- **Terep:** 10×10 m-es csempék véletlen helyzetű függőleges hengerekkel
  (`MeshRepeatedCylindersTerrainCfg`). A csempe közepén egy 3×3 m-es oszlopmentes spawn platform van.
  10 nehézségi szint (sor): 4 db 0,15 m sugarú oszloptól 30 db 0,30 m sugarú oszlopig, 20 variáns szintenként.
- **Command:** célpozíció a csempén (`GoalPositionCommand`), epizódonként egyszer mintavételezve,
  mindig legalább 0,7 m-re bármely oszloptól (flat patch sampling). A policy a célt a robot
  (yaw-igazított) bázis-koordinátarendszerében kapja (x, y). Vizualizáció: piros gömb a célon, zöld nyíl minden
  robot fölött a célja felé; a kamera által követett env 1 célja és nyila sárga.
- **Szenzor:** 360°-os síklidar (`RayCaster`, 72 sugár, 5 m), a base link felett 0,15 m-rel.
- **Megfigyelés:** bázis lin/ang sebesség, cél (x, y), hátralévő idő, 72 lidar-távolság, 4 kerékfordulat, előző action.
- **Action:** a 4 kerék fordulat-alapjele, amit egy nyomatékkorlátos DC-motor modell követ, így a csúszás
  ott jelenik meg, ahol a valóságban: ha a nyomatékigény meghaladja a tapadást.
- **Reward** — Rudin et al. 2022, [arXiv:2209.12827](https://arxiv.org/abs/2209.12827) alapján. Epizód T = 12 s.
  | Tag | Képlet | Súly |
  |---|---|---|
  | fő jutalom, (1) | `1/T_r · 1/(1+d²)`, csak az utolsó `T_r` = 2 s-ban | 10 (epizódösszeg ≤ 10) |
  | exploration bias, (3) | `cos(v, cél irány)`; automatikusan kikapcsol, ha a fő jutalom eléri a max. 50%-át | 0,5 |
  | stalling, (4) | 1, ha `v < 0,1 m/s` és `d > 0,5 m` | −0,5 |
  | kerék-nyomaték² (a cikk joint torque-ja) | `Σ τ²` | −2e-4 |
  | kerékgyorsulás² (joint acceleration) | `Σ ω̇²` | −1e-6 |
  | kerékcsúszás² (a lábgyorsulás megfelelője) | `Σ (R·ω − gördülési sebesség)²` | −0,5 |
  | akcióváltozás² | `‖a − a₋₁‖²` | −0,01 |
  | oszlop-közelség | lineáris, ha a rés a robot téglalapja és az oszlop között < 0,25 m | −1 |
  | ütközés / felborulás | terminál, így a fő jutalom is elvész | −100 |
  | alváz roll/pitch sebesség² | | −0,05 |
  A time-out nem truncation (nincs value-bootstrapping), mert a hátralévő időt a policy és a critic is látja;
  PPO 48 lépés/env/iteráció, 2000 iteráció (mint a cikkben).
- **Termináció:** ütközés, ha a lidar szerint egy oszlop 3 cm-nél közelebb van a robot téglalapjához
  (`FOOTPRINT_HALF_EXTENTS`, 0,66×0,60 m a kerekekkel); felborulás; timeout (12 s).
- **Curriculum:** célba ért (≤ 0,5 m) → nehezebb szint; ütközött vagy > 2 m-re maradt → könnyebb szint.
- **Domain randomization:** görgő-talaj súrlódás 0,5–1,1 (statikus) / 0,4–1,0 (dinamikus).

## A görgős mecanum robot

```bash
python scripts/tools/generate_mecanum_robot.py      # -> assets/robots/mecanum/robot.urdf (62 link)
```

- **Kerék:** a [fuji_mecanum](https://github.com/DaiGuard/fuji_mecanum) FUJI FM202 kerék geometriája
  (45°-os görgők, görgőtengely 0,0895 m-re a keréktengelytől, 13 mm görgősugár → 102,5 mm burkoló sugár).
  A görgők szabadon forgó jointok; a TIAGo-integrációhoz ([tiago_isaac](https://github.com/AIS-Bonn/tiago_isaac))
  hasonlóan minden görgő 6 gömbön keresztül ütközik, a gömbök sugara úgy van választva, hogy mindegyik érintse
  a kerék hengeres burkolóját (hordóprofil `r(s) = R - sqrt(Rc² + s²cos²45°)`). A keréktárcsának nincs ütközője,
  az önütközés ki van kapcsolva → csak görgő–talaj kontaktus létezik.
- **Robot:** 0,6×0,4 m-es alváz (18 kg, összesen 24,7 kg), lx = 0,22 m, ly = 0,26 m, "X" görgőelrendezés,
  lengő első tengely (3 pontos, statikailag határozott alátámasztás).
- **Motor:** DC motor modell (20 Nm álló nyomaték, 10 Nm folyamatos, 30 rad/s üresjárati fordulat),
  sebességszabályzó d = 2 Nm·s/rad, reflektált rotor-inercia 0,01 kg·m².
- **Kontaktus:** rugalmas (poliuretán) görgő-kontaktus k = 2·10⁵ N/m, c = 1000 N·s/m.

Eltérések a FUJI keréktől / TIAGo-tól, mind a validáció eredménye:

| Eltérés | Ok |
|---|---|
| 14 görgő 15 helyett (60 mm hosszú görgők az átfedés megtartásához) | a PhysX GPU artikuláció > 64 linknél sok env-nél csendben felrobban (64 env-nél pontosan az első 32 robot kilövődik); 15 görgővel 66 link lenne |
| lengő első tengely | merev 4 kerekű alváz merev kontaktuson az átlói között billeg; átlós menetnél a két hajtott kerék leemelkedett (0,25× gyorsulás) |
| rugalmas kontaktus | merev kontaktussal a merev kerék a görgőátmeneteknél pattog (a lépések akár 33%-ában terheletlen kerék) |
| görgőcsillapítás 1e-5 (Gazebo-modellben 1e-3) | 12 mm-es görgőn az 1e-3 ~2 N gördülési ellenállás, torzította a kinematikát |
| motor-armatúra 0,01 kg·m² | nélküle az explicit motormodell instabil (`d·dt/I > 2`), a kerék bang-bang oszcillált |

A compliance-t a PhysX csak a **talaj** anyagán veszi figyelembe, ezért a taskok `mecanum.ground_material()`-t
használnak a terepre és `mecanum.ROBOT_MATERIAL`-t alapértelmezett anyagnak.

### Validáció

```bash
python scripts/tools/validate_mecanum_wheels.py --headless --physics_hz 360 --out logs/validation/hz360_it8.json
python scripts/tools/summarize_validation.py logs/validation/*.json
```

Minden env egy tesztesetet futtat (kétszeres példányban, 72 env), sík talajon μ = 0,8-cal, fizikai lépésenként naplózva
(nyers adatok: `.npz`). Eredmények 360 Hz / 8 iteráció mellett:

| Szint | Teszt | Eredmény |
|---|---|---|
| 1 geometria | kerekség (offline, gömbök burkolója) | 0,10 mm hullámzás |
| | alváz magasság-hullámzás lassú menetnél | 0,18 mm |
| | kontaktus-folytonosság (minden keréknek mindig van ≥ 0,5 N-os görgője) | 0% kimaradás |
| | nem-talaj kontaktus / alváz kontaktus | 0 N / 0 N; terhelés/súly = 1,00 |
| 2 kinematika | előre / oldalra / átló / forgás relatív hiba (0,3 és 0,8 m/s, ±) | 1,8% / 3,7% / 2,6% / 3,4% |
| | keresztcsatolás, szimmetria (+/−) | 0,35%, 0,13% |
| | átlón a 0-ra parancsolt kerekek nyomatéka | 0,07 Nm (hajtott: 0,40 Nm) |
| 3 tapadás | max. gyorsulás / előrejelzés (μg/√2 előre-oldalra, μg/2 átlósan) | 1,01 / 1,00 / 0,995 (μ = 0,5-nél 1,03 / 0,98 / 1,00) |
| | csúszás kezdete (rámpa) | átló: 0,9–1,1× előrejelzés; előre fokozatos (terhelésátvitel) |
| 4 numerika | 120 / 240 / 360 / 480 Hz, 4 / 8 / 16 iteráció | 360 Hz-től konvergál (tapadás 0,99–1,02, hullámzás 0,17–0,18 mm); 120 Hz: 0,4 mm, 0,93× |
| | rezgésspektrum (8 Hz felett) | ≥ 360 Hz: minden csúcs a görgőátmenet-frekvencia egész többszöröse (5× = 6 gömb közti átadás); 120 Hz: nem egész többszörösök → numerikus |
| 5 kereszt | ideális kinematikai modell vs. Isaac | lásd 2. szint; Gazebo nincs telepítve, az a kereszt-szimulátoros összevetés következő lépése |

A kritériumokat (`summarize_validation.py` `CRITERIA`) ideális kinematikához választottam, nem valódi robothoz;
valódi hardveren a súrlódás és csillapítás identifikálása a következő lépés.

## Használat

```bash
conda activate env_isaaclab
cd /home/peter.varadi@egroup.hu/IsaacLab/mecanum_ws

# regisztrált taskok
python scripts/list_envs.py

# gyors ellenőrzés tanítás nélkül (GUI)
python scripts/zero_agent.py --task Mecanum-Velocity-Flat-v0 --num_envs 4

# tanítás
python scripts/rsl_rl/train.py --task Mecanum-Velocity-Flat-v0 --headless
python scripts/skrl/train.py   --task Mecanum-Velocity-Flat-v0 --headless

# a betanított policy visszajátszása (legutóbbi checkpoint a logs/ alól)
python scripts/rsl_rl/play.py --task Mecanum-Velocity-Flat-Play-v0 --num_envs 16
```

A konfig bármely mezője felülírható Hydra argumentummal, pl. más URDF kipróbálásához
(a `robots/mecanum.py` konstansai legyenek szinkronban vele):

```bash
python scripts/rsl_rl/train.py --task Mecanum-Velocity-Flat-v0 --headless \
    env.scene.robot.spawn.asset_path=/abs/path/robot.urdf
```

## Navigációs tanítás wandb-vel

Egyszer be kell jelentkezni a wandb-be, utána indítható a tanítás:

```bash
conda activate env_isaaclab
cd /home/peter.varadi@egroup.hu/IsaacLab/mecanum_ws
python -m wandb login        # egyszer, a wandb API kulccsal

python scripts/rsl_rl/train.py --task Mecanum-Navigation-Pillars-v0 --headless --num_envs 1536 \
    --video --video_length 500 --video_interval 4800 \
    --logger wandb --log_project_name mecanum-navigation
```

- `--logger wandb --log_project_name ...`: a tanítási adatokat (és a felvett videókat) wandb-be menti.
- `--video --video_length 500 --video_interval 4800`: 100 iterációnként egy 10 s-os videó (az env 1 robotját követve).
- `--num_envs 1536`: a GPU-t más folyamatok is használják (~7,5 GB), 1536 env + videó ~8 GB; 2048 + videó már nem fér el.
- Folytatás: `--resume --load_run <run_mappa>` (új tanítás kell viszont, ha a régi checkpoint a mostani, bővített
  megfigyelés előtti — a hátralévő idő új obs).

## Kiértékelés és reward-tuning

```bash
# egy checkpoint kiértékelése (determinisztikus policy, minden nehézségi szinten): siker, ütközés és iránya,
# sebességprofil az epizód alatt, szintenkénti bontás -> <run>/eval_model_<N>.json
python scripts/rsl_rl/evaluate_navigation.py --checkpoint logs/rsl_rl/mecanum_navigation_pillars/<run>/model_900.pt

# súlyok / hiperparaméterek keresése: minden próba = tanítás (korai leállással) + kiértékelés, rangsor a végén
python scripts/tools/tune_rewards.py --study round1 --wandb --video   # a TRIALS listája, prioritási sorrendben
python scripts/tools/tune_rewards.py --study rnd1 --random 8          # 8 véletlen kombináció a SEARCH_SPACE-ből
python scripts/tools/tune_rewards.py --study round1 --report          # csak a rangsor
```

- A próbák Hydra-override-ok (pl. `env.rewards.wheel_slip_l2.weight`, `agent.algorithm.gamma`,
  `env.episode_length_s`, `env.commands.goal_pose.min_distance`), a szkript tetején (`TRIALS`, `SEARCH_SPACE`).
- Alapból 3072 env (videóval ~10,5 GB VRAM, ~15 s/iteráció), legfeljebb 500 iteráció, 25 iterációnként checkpoint.
  4096 env + videó az Omniverse renderelőben elszáll (cubric assertion → CUDA illegal address); videó nélkül a 4096 is fut.
- **Korai leállás:** a 150. iterációtól 25 iterációnként összeveti az utolsó 75 iteráció átlagos mean rewardját
  és curriculum-szintjét az azt megelőző 75-ével; ha kétszer egymás után < 3% a reward-javulás és < 0,2 a
  szintemelkedés, leállítja a tanítást és a legutóbbi checkpointot értékeli ki (`--no_early_stop`, `--es_*`).
- A kiértékelés minden próbánál ugyanazon a célkiosztáson fut (min. 1 m start–cél távolság), csak az epizódhossz
  jön a próbából. Eredmények: `logs/tuning/<study>/results.jsonl`, `leaderboard.md`; megszakítás után ugyanazzal a
  paranccsal folytatódik (a kész próbákat átugorja).
- Pontszám: `siker − 0,5·ütközés − 0,01·medián odaérési idő [s]`.
- `--wandb`: group = study neve, a kiértékelés `eval/*` összesítőként és `study:`/`early_stopped` tagként a runhoz
  kerül; előtte `export WANDB_USERNAME=varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem`.

## Büntetések lépésenkénti visszakapcsolása (sík terep)

```bash
WANDB_USERNAME=varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem setsid nohup \
  /home/bence.farkas@egroup.hu/miniconda3/envs/env_isaaclab/bin/python -u scripts/tools/staged_penalties.py \
  --study flat_penalties --wandb --video > logs/staged/flat_penalties_study.log 2>&1 < /dev/null &
```

Kiindulás: a csak pozitív jutalommal (`final_position` + `exploration_bias`) tanított sík terepes policy
(`*flat_task_and_bias/model_499.pt`). A szkript megvárja ennek a végét, kiértékeli (kapu: ≥ 85% siker), majd
sorban egyenként bekapcsolja a büntetéseket: `wheel_torque_l2` → `action_rate_l2` → `wheel_acc_l2` → `stalling` →
`wheel_slip_l2` → `action_l2`. Minden lépésben az előzőleg elfogadott policyből folytat (`--resume`, 150 iteráció),
a súlyokat a legerősebbtől a leggyengébbig próbálja, és az első olyat fogadja el, amelynél a siker legfeljebb 3
százalékponttal, az odaérési idő legfeljebb 1 s-mal romlik. Ha egy lépés egyik súlya sem felel meg, megáll. Eredmény:
`logs/staged/<study>/report.md` (a hatás-metrikákkal: nyomaték², akcióváltás², gyorsulás², csúszás², pörgés a célban).
A büntetések a konfigban 0 súllyal szerepelnek (a reward manager kihagyja őket), a szkript Hydra-override-dal kapcsolja be.

## Videó a tanításról

A `--video` kapcsoló headless módban is működik (automatikusan bekapcsolja a kamerákat). A kamera az
env 0 robotját követi (`viewer` beállítás a env configban). A videók ide kerülnek:
`logs/rsl_rl/<experiment>/<run>/videos/train/rl-video-step-<N>.mp4` (play-nél `videos/play/`).

```bash
# minden 3000. env lépésnél (~125 iteráció) egy 500 lépéses (10 s) videó
python scripts/rsl_rl/train.py --task Mecanum-Navigation-Pillars-v0 --headless \
    --video --video_length 500 --video_interval 3000

# a betanított policy videója
python scripts/rsl_rl/play.py --task Mecanum-Navigation-Pillars-Play-v0 --headless --video --video_length 1000
```

## Livestream a saját gépre (WebRTC)

1. A saját gépedre telepítsd az **Isaac Sim WebRTC Streaming Client**-et
   (https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html → Latest Release).
2. A szerveren indítsd `--livestream 2` kapcsolóval (privát hálózat / LAN; a `--headless` automatikus):

   ```bash
   python scripts/rsl_rl/play.py --task Mecanum-Navigation-Pillars-Play-v0 --livestream 2
   # vagy tanítás közben (a renderelés lassítja a tanítást, ezért kevesebb env ajánlott)
   python scripts/rsl_rl/train.py --task Mecanum-Navigation-Pillars-v0 --num_envs 1024 --livestream 2
   ```

3. Várd meg, amíg a logban megjelenik, hogy az app betöltött (első indításkor ez percekig tarthat), majd
   a kliensben a szerver IP-jére csatlakozz (`10.100.50.104`).

Hálózati követelmény: a szerver **TCP 49100** és **UDP 47998** portja legyen elérhető a gépedről
(ha VPN-en vagy tűzfal mögül nem megy, a videófelvétel mindig működik). NAT/publikus IP mögött
`--livestream 1` kell, és előtte `export PUBLIC_IP=<a szerver publikus IP-je>`.
Egyszerre csak egy kliens csatlakozhat.

A logok és checkpointok a `logs/rsl_rl/<experiment_name>/` (ill. `logs/skrl/`) alá kerülnek.
A csomag editable módban telepítve van az `env_isaaclab` környezetbe
(`pip install -e source/mecanum_ws`), így a forráskód módosításai azonnal érvényesek.
