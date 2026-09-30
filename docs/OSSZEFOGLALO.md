# Mecanum robot end-to-end navigációja megerősítéses tanulással — összefoglaló

Állapot: 2026-09-30. Kódállapot: a `main` ág (<https://github.com/Kemenylegeny/Mecanum-TDK>); a tanítások részletes
naplója: [`EXPERIMENTS.md`](../EXPERIMENTS.md); wandb-projekt:
<https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation>.

## Tartalom

1. [A feladat röviden](#1-a-feladat-röviden)
2. [A mecanum kerék modellje](#2-a-mecanum-kerék-modellje)
3. [A robot URDF-je](#3-a-robot-urdf-je)
4. [A fizikai modell validációja](#4-a-fizikai-modell-validációja)
5. [Szimulációs beállítások](#5-szimulációs-beállítások)
6. [A környezet: terep, cél, lidar](#6-a-környezet-terep-cél-lidar)
7. [Megfigyelések](#7-megfigyelések)
8. [Akciók: kerékfordulat vagy testsebesség? Az inverz kinematika](#8-akciók-kerékfordulat-vagy-testsebesség-az-inverz-kinematika)
9. [Jutalmak és büntetések, és hogyan számolódnak](#9-jutalmak-és-büntetések-és-hogyan-számolódnak)
10. [Epizód vége (termináció)](#10-epizód-vége-termináció)
11. [Curriculum](#11-curriculum)
12. [A neurális háló](#12-a-neurális-háló)
13. [PPO: konfiguráció és egy iteráció menete](#13-ppo-konfiguráció-és-egy-iteráció-menete)
14. [A tanítás története](#14-a-tanítás-története)
15. [Hibák és tanulságok](#15-hibák-és-tanulságok)
16. [Jelenlegi állapot és javasolt következő lépések](#16-jelenlegi-állapot-és-javasolt-következő-lépések)
17. [Eszközök és fájlok](#17-eszközök-és-fájlok)

---

## 1. A feladat röviden

Egy négy mecanum kerekes, TIAGo-szerű mobil robot szimulációban (Isaac Sim 5.1 / Isaac Lab 2.3.2) tanul meg
**end-to-end** eljutni egy célpontig függőleges oszlopok (hengerek) között. „End-to-end”: nincs térkép, pályatervező
vagy külön szabályzó — a neurális háló közvetlenül a szenzoradatokból (lidar, saját sebesség, cél helyzete) adja ki a
négy kerék fordulat-alapjelét, 51 Hz-en.

- Tanítás: PPO (RSL-RL), 3072 párhuzamos robot egy GPU-n (RTX 5070 Ti, 16 GB), iterációnként ~22 s.
- Jutalom: Rudin et al. 2022 (*Advanced Skills by Learning Locomotion and Local Navigation End-to-End*,
  [arXiv:2209.12827](https://arxiv.org/abs/2209.12827)) szerint csak az epizód végén jár, a cél távolsága alapján; a
  mozgás minőségét büntetések formálják (nyomaték, akcióváltás, kerékcsúszás).
- Nehézség: curriculum, 10 szint — 4 vékony oszloptól 30 vastag oszlopig.

---

## 2. A mecanum kerék modellje

Generátor: [`scripts/tools/generate_mecanum_robot.py`](../scripts/tools/generate_mecanum_robot.py); a paraméterek a
[`assets/robots/mecanum/wheel_geometry.json`](../assets/robots/mecanum/wheel_geometry.json)-ban.

**Kiindulás:** a FUJI FM202-205-15U mecanum kerék ([fuji_mecanum](https://github.com/DaiGuard/fuji_mecanum)):
45°-os görgők, a görgőtengely 0,0895 m-re a keréktengelytől, 13 mm görgősugár középen → a kerék burkoló sugara
R = 0,0895 + 0,013 = **0,1025 m**.

**Görgők mint szabad jointok:** minden görgő egy külön merev test, egy passzív (nem hajtott) forgó jointtal a
keréktárcsán, 1e-5 N·m·s/rad csapágycsillapítással. A görgő tengelye a kerék érintőjéhez képest 45°-ban áll. A
görgőtengely iránya a θ szöghelyzetű görgőnél (θ = 0 alul):
`u = cos(γ)·t(θ) + h·sin(γ)·y`, ahol `t(θ) = (cos θ, 0, sin θ)` az érintő, γ = 45°, `h = ±1` a kerék
„kezessége” (bal/jobb menetes).

**Ütközési geometria — gömbsor:** a TIAGo Isaac-integrációjához ([tiago_isaac](https://github.com/AIS-Bonn/tiago_isaac))
hasonlóan minden görgő **6 gömbön** keresztül ütközik (henger/mesh ütközők a PhysX-ben pontatlanok és lassúak). A
gömbök a görgő tengelye mentén ülnek, és a sugaruk úgy van választva, hogy mindegyik pontosan érintse a kerék
hengeres burkolóját (hordóprofil): `r(s) = R − sqrt(Rc² + s²·cos²45°)`, ahol `s` a görgő mentén mért távolság,
`Rc` = 0,0895 m. Így a gömbsugarak 10,5 / 12,1 / 12,9 / 12,9 / 12,1 / 10,5 mm. A keréktárcsának nincs ütközője, az
önütközés ki van kapcsolva — a kerék **csak görgő–talaj kontaktuson** keresztül érintkezik.

**Kerekség (offline ellenőrzés):** a gömbök burkolójának sugara 0,10240–0,10250 m között változik, a hullámzás
**0,10 mm**; görgőosztásonként 3 görgőátadás van, a lefedés folytonos.

**Eltérések a FUJI keréktől (szimulátor-kényszer):**

| Eltérés | Ok |
|---|---|
| 14 görgő 15 helyett (60 mm hosszú görgők az átfedés megtartásához) | a PhysX GPU artikuláció > 64 linknél sok env mellett figyelmeztetés nélkül felrobban (64 env-nél pontosan az első 32 robot kilövődött); 15 görgővel 1 + 1 + 4 + 60 = 66 link lenne |
| görgőcsillapítás 1e-5 (a Gazebo-modellekben 1e-3) | a 12 mm-es görgőn az 1e-3 már ~2 N gördülési ellenállás 0,3 m/s-nál, és látványosan torzította a kinematikát |

---

## 3. A robot URDF-je

A generátor egy teljes URDF-et ír: [`assets/robots/mecanum/robot.urdf`](../assets/robots/mecanum/robot.urdf) (a
görgő vizuális mesh-e: `meshes/roller.stl`). A robot-oldali konstansok
([`robots/mecanum.py`](../source/mecanum_ws/mecanum_ws/robots/mecanum.py)) ezzel szinkronban vannak.

| Elem | Érték |
|---|---|
| Alváz | 0,6 × 0,4 × 0,12 m doboz, 18 kg (a robot összesen ~24,7 kg) |
| Kerékelrendezés | „X” (felülnézetből a görgők X-et rajzolnak), lx = 0,22 m (fél tengelytáv), ly = 0,26 m (fél nyomtáv) |
| Keréktárcsa | r = 0,075 m, szélesség 64,4 mm, 1 kg |
| Görgő | 30 g, 14 db/kerék |
| Lengő első tengely | a két első kerék egy közös, x körül ±0,12 rad-ot (~3 cm a kerekeknél) szabadon lengő tengelyen, 1 kg, csillapítás 1 N·m·s/rad |
| Linkek | 1 alváz + 1 tengely + 4 tárcsa + 56 görgő = **62** (≤ 64) |
| Befoglaló téglalap (kerekekkel) | 0,66 × 0,60 m (`FOOTPRINT_HALF_EXTENTS = (0,33, 0,30)`) |

**Miért lengő tengely?** Egy merev, négykerekű alváz merev kontaktuson statikailag határozatlan: az átlói között
billeg, és átlós menetnél épp a két hajtott kerék emelkedett le (a gyorsulás a várt 0,25-szöröse lett). A lengő
tengellyel hárompontos, statikailag határozott az alátámasztás.

**Motor (kerékhajtás):** Isaac Lab `DCMotorCfg` — a policy fordulat-alapjelét egy sebességszabályzó követi
(τ = d·(ω_alapjel − ω), d = 2 N·m·s/rad), a nyomatékot egy lineáris nyomaték–fordulat görbe korlátozza (álló nyomaték
20 N·m, üresjárati fordulat 30 rad/s), és egy folyamatos korlát (10 N·m). A 10 N·m jóval a tapadási határ (~3 N·m
kerekenként μ = 0,8-nál) felett van, így **a kerék megcsúszhat**, ahogy a valóságban. Reflektált rotor-inercia
(armatúra) 0,01 kg·m² — nélküle az explicit (fizikai lépésenként egyszer számolt) motormodell instabil
(`d·dt/I > 2` esetén bang-bang oszcilláció). A max. parancsolt kerékfordulat 20 rad/s (≈ 2 m/s).

**Kontaktus:** rugalmas (compliant) görgő–talaj kontaktus, k = 2·10⁵ N/m, c = 1000 N·s/m (poliuretán görgő, ~0,3 mm
benyomódás). Merev kontaktussal a merev kerék a görgőátmeneteknél pattogott (a lépések akár 33%-ában terheletlen
kerék). A PhysX a compliance-t csak a **talaj** anyagán veszi figyelembe, ezért a terep anyaga
`mecanum.ground_material()`, a robot ütközői pedig a szimuláció alapértelmezett anyagát kapják
(`ROBOT_MATERIAL`, súrlódás 1,0, szorzásos kombinálás).

---

## 4. A fizikai modell validációja

Nyílt hurkú tesztek RL nélkül ([`validate_mecanum_wheels.py`](../scripts/tools/validate_mecanum_wheels.py),
[`summarize_validation.py`](../scripts/tools/summarize_validation.py)), sík talajon, μ = 0,8, 360 Hz / 8 iteráció:

| Szint | Teszt | Eredmény |
|---|---|---|
| geometria | alváz magasság-hullámzás lassú menetnél | 0,18 mm |
| | kontaktus-folytonosság (mindig van ≥ 0,5 N-os görgő) | 0% kimaradás |
| kinematika | előre / oldalra / átló / forgás relatív hiba | 1,8% / 3,7% / 2,6% / 3,4% |
| | keresztcsatolás, +/− szimmetria | 0,35%, 0,13% |
| tapadás | max. gyorsulás / elméleti (μg/√2 előre-oldalra, μg/2 átlósan) | 1,01 / 1,00 / 0,995 |
| numerika | 120 / 240 / 360 / 480 Hz | 360 Hz-től konvergál; 120 Hz-en a rezgéscsúcsok nem a görgőátadás többszörösei → numerikus zaj |

A kinematikai tesztek a [8. fejezet](#8-akciók-kerékfordulat-vagy-testsebesség-az-inverz-kinematika) inverz
kinematikájával számolt ideális mozgással hasonlítják össze a szimulációt.

---

## 5. Szimulációs beállítások

| Beállítás | Érték | Megjegyzés |
|---|---|---|
| fizikai lépés | 1/360 s | a validáció szerint ez alatt a kontaktus-zaj numerikus |
| decimation | 7 | a policy 360/7 = **51,4 Hz**-en dönt (lépésköz dt = 7/360 = 19,4 ms) |
| solver | 8 pozíció- / 1 sebesség-iteráció | |
| `gpu_max_rigid_patch_count` | 2¹⁹ = 524 288 | alapértéke 163 840; lásd [15. fejezet](#15-hibák-és-tanulságok) (PhysX overflow) |
| domain randomization | görgő–talaj súrlódás: statikus 0,5–1,1, dinamikus 0,4–1,0 (induláskor, 64 csoportban) | a valódi súrlódás ismeretlen |
| envek | 3072 (videóval ~11 GB VRAM) | 4096 + videó determinisztikusan összeomlik a renderelőben |

---

## 6. A környezet: terep, cél, lidar

Konfiguráció: [`navigation_env_cfg.py`](../source/mecanum_ws/mecanum_ws/tasks/manager_based/mecanum_navigation/navigation_env_cfg.py).

**Terep:** 10 × 20 csempe, mindegyik 10 × 10 m (5 m szegéllyel). A **sor a nehézségi szint** (curriculum):
a 0. szinten 4 db 0,15 m sugarú oszlop, a 9. szinten 30 db 0,30 m sugarú oszlop (1,5 m magasak, közte lineárisan
nő a szám és a sugár); szintenként 20 véletlen variáns (a rács oszlopai). A csempe közepén egy 3 × 3 m-es oszlopmentes
platform van, itt indul a robot.

**Indulás (reset):** véletlen pozíció a csempe közepén ±0,3 m-en belül, véletlen irány (yaw ±180°).

**Cél (`GoalPositionCommand`):** epizódonként egyszer, a csempe ±4,5 m-es területén, olyan pontok közül, amelyek
legalább 0,7 m-re vannak minden oszloptól (flat patch sampling). A medián kezdeti céltávolság ~3,4–3,5 m. A
kiértékelésnél a cél legalább 1 m-re van (`min_distance = 1,0`). Vizualizáció: piros gömb a célon, zöld nyíl a robot
fölött a cél felé; a kamera által követett 1-es env célja és nyila sárga.

**Lidar:** síklidar (Isaac Lab `RayCaster`), 72 sugár 5°-onként (360°), 5 m hatótáv, a base link felett 0,15 m-rel,
csak a robot yaw-jával forog (a dőlésével nem). Csak a terep mesht (talaj + oszlopok) látja, a többi robotot nem.

**Epizódhossz:** jelenleg **9 s** (a pillaros tanításoknál korábban 12, majd 7 s). A flat (oszlop nélküli)
változatnál 7 s.

**Megjegyzés — sok robot egy csempén:** 3072 robot jut 200 csempére, a curriculum elején (0–2. szint, 60 csempe)
~50 robot egy csempén, ugyanonnan indulva. A különböző envek robotjai nem ütköznek egymással (és a lidar sem látja
őket), ezért a videón egy „kupacnak” látszanak — ez fizikailag rendben van.

---

## 7. Megfigyelések

A policy bemenete egy 89 elemű vektor (a critic ugyanezt kapja):

| Tag | Dim. | Tartalom | Zaj (tanításkor, egyenletes) |
|---|---|---|---|
| `base_lin_vel` | 3 | test lineáris sebessége a test koordináta-rendszerében [m/s] | ±0,05 |
| `base_ang_vel` | 3 | test szögsebessége [rad/s] | ±0,1 |
| `goal_position` | 2 | a cél (x, y) a robot yaw-igazított koordináta-rendszerében [m] | ±0,05 |
| `time_left` | 1 | hátralévő idő az epizódhossz arányában (1 → 0) | – |
| `lidar` | 72 | lidar-távolságok / 5 m (1 = nincs akadály 5 m-en belül) | ±0,01 |
| `wheel_vel` | 4 | kerékfordulatok / 20 rad/s | ±0,02 |
| `actions` | 4 | az előző akció | – |

A `time_left` azért kell, mert a fő jutalom csak az epizód végén jár: a policynek és a criticnek tudnia kell, mennyi
ideje van hátra (ezért nem is kell a timeoutnál „bootstrapolni”, lásd [10. fejezet](#10-epizód-vége-termináció)). A
bemenetet a háló futó átlaggal/szórással normalizálja (empirical normalization).

---

## 8. Akciók: kerékfordulat vagy testsebesség? Az inverz kinematika

### Mit ad ki a jelenlegi policy?

**Közvetlenül a négy kerék fordulat-alapjelét** (`Mecanum-Navigation-Pillars-v0`, `ActionsCfg.wheel_vel`,
Isaac Lab `JointVelocityAction`): az akció 4 szám, sorrendben bal első, jobb első, bal hátsó, jobb hátsó kerék;
`ω_alapjel = 20 rad/s · a`, ±20 rad/s-ra vágva. Az alapjelet a DC motor modell követi. Az inverz kinematikát (IK)
tehát a policynek magának kell megtanulnia.

Van egy **testsebességes változat** is (`Mecanum-Navigation-Pillars-IK-v0`, `IkActionsCfg.base_vel`,
[`MecanumBaseVelocityAction`](../source/mecanum_ws/mecanum_ws/tasks/manager_based/mecanum_velocity/mdp/actions.py)):
a policy 3 számot ad ki — a test sebességét (vx, vy, ωz), max. 1 m/s, 1 m/s, 1,5 rad/s —, és az IK számolja át
kerékfordulattá. **Ezt a változatot eddig egyszer sem tanítottuk** (a naplóban nincs `mecanum_navigation_pillars_ik`
run), minden eddigi eredmény a közvetlen kerékfordulatos policyé.

### Az inverz kinematika levezetése

A mecanum kerék kerületén 45°-ban álló, szabadon forgó görgők vannak. A kerék a talajon két irányban tud mozogni:
a kerék forgásával (a kerék síkjában), és a görgő forgásával (a görgő tengelyére merőlegesen) — de **a görgő
tengelye mentén nem**: ebben az irányban csak csúszni tudna. Ez az egyetlen kényszer keréknként.

1. Ha a test sebessége (vx, vy) és szögsebessége ωz, akkor az (xᵢ, yᵢ) helyen lévő kerék középpontjának sebessége
   `vᵢ = (vx − ωz·yᵢ,  vy + ωz·xᵢ)`.
2. A kerék forgása a kontaktpontban `ωᵢ·R` sebességet ad a kerék síkjában (x irány). A talajjal érintkező görgő
   tengelye a generátor szerint `uᵢ = (cos 45°, hᵢ·sin 45°)`, hᵢ = ±1 a kerék kezességétől függően.
3. Csúszásmentesség: a kontaktpont relatív sebességének nincs komponense a görgő tengelye mentén:
   `(vᵢ − ωᵢ·R·x̂) · uᵢ = 0`  ⇒  `ωᵢ·R = vᵢₓ + hᵢ·tan45°·vᵢᵧ = vᵢₓ + hᵢ·vᵢᵧ`.
4. Behelyettesítve a négy kerékre (bal első (lx, ly), h = −1; jobb első (lx, −ly), h = +1; bal hátsó (−lx, ly), h = +1;
   jobb hátsó (−lx, −ly), h = −1), „X” elrendezésnél:

```
ω_BE = (vx − vy − (lx + ly)·ωz) / R
ω_JE = (vx + vy + (lx + ly)·ωz) / R
ω_BH = (vx + vy − (lx + ly)·ωz) / R
ω_JH = (vx − vy + (lx + ly)·ωz) / R        R = 0,1025 m, lx + ly = 0,48 m
```

Ez egy 4 × 3-as mátrix (`J`, `ω = J·[vx, vy, ωz]ᵀ`). A **direkt kinematika** a pszeudoinverze:

```
vx = R/4 · ( ω_BE + ω_JE + ω_BH + ω_JH)
vy = R/4 · (−ω_BE + ω_JE + ω_BH − ω_JH)
ωz = R / (4·(lx + ly)) · (−ω_BE + ω_JE − ω_BH + ω_JH)
```

**A 4. dimenzió — a „nulltér”:** négy kerék van, de a test síkbeli mozgása csak 3 szabadsági fokú. A kerékfordulatok
terében van egy irány, ami semmilyen mozgást nem ad: `n = (1, 1, −1, −1)` (az első kerekek előre, a hátsók hátra
forognak — egymás ellen dolgoznak). Ha a kerékparancsnak van ilyen komponense, az csak csúszást, nyomatékot és
energiaveszteséget okoz. A `wheel_slip_l2` büntetés pontosan ezt méri: a kerék kerületi sebessége
(`R·ω`) mennyiben tér el attól, amit a test tényleges mozgása az IK szerint megkövetel
([`rewards.py`](../source/mecanum_ws/mecanum_ws/tasks/manager_based/mecanum_navigation/mdp/rewards.py)).

**Honnan van?** Ez a mecanum platformok standard kinematikai modellje: Muir & Neuman, *Kinematic modeling for
feedback control of an omnidirectional wheeled mobile robot* (ICRA 1987); tankönyvi levezetés: Lynch & Park,
*Modern Robotics* (2017), 13.2 fejezet (omnidirekcionális kerekes robotok); a 4 kerekes „X” elrendezésre pl. Taheri,
Qiao & Ghaeminezhad, *Kinematic model of a four mecanum wheeled mobile robot* (IJCA, 2015). A mi kódunkban a
kerékkezességet és az előjeleket a generált URDF geometriájából vezettük le (a generátor docstringje), és a
validáció kinematikai tesztjei (1,8–3,7% hiba) ezzel az IK-val számolt ideális mozgáshoz mérnek.

### Melyikkel érdemesebb tanítani?

| | Közvetlen kerékfordulat (4D, jelenlegi) | Testsebesség + IK (3D) |
|---|---|---|
| Akciótér | 4D, benne a mozgást nem adó nulltér | 3D, minden akció értelmes mozgás |
| Mit kell megtanulni | a navigációt **és** a kerékkinematikát | csak a navigációt |
| Felfedezés | a zaj egy része a nulltérbe esik → kerekek egymás ellen, csúszás, forgás | a zaj közvetlenül a mozgás irányát/sebességét változtatja |
| Csúszás | a policynek kell kerülnie (ezért kellett slip-büntetés) | kinematikailag konzisztens; csak dinamikus csúszás (tapadási határ) marad |
| Lehetőség | kihasználhatja a dinamikát (pl. tapadási határ, motor-telítés kerekenként) | a kerekenkénti finomságot az IK „elveszi” |
| Valódi robot | a policy a motor-szabályzókat közvetlenül hajtja; a sim-to-real rés nagyobb (kerék-, görgő-, motor-modell) | a szokásos `cmd_vel` interfész; a kerékszabályzás a robot saját feladata |
| A büntetések jelentése | akcióváltás = kerékparancs-ugrás | akcióváltás = test-gyorsulás (fizikailag szemléletesebb) |

**Várakozásom:** a testsebességes változat gyorsabban és simábban tanul, kevesebb forgással és csúszással —
pontosan azok a hibák, amiket a jelenlegi policynél látunk (sok forgás a célnál, csak ~43–59%-ban előre haladás, a
slip-büntetés szükségessége), részben a 4D akciótér következményei. A közvetlen kerékfordulatnak akkor lehet előnye,
ha a policynek a tapadási határ közelében kell dolgoznia. Ez nem „nem számít” kérdés: érdemes a kettőt ugyanazzal a
jutalommal összehasonlítani — a kód megvan (`Mecanum-Navigation-Pillars-IK-v0`), csak egy tanítás. Megjegyzés: az IK
változatnál a max. testsebesség 1 m/s (a jelenlegi policy ~0,6 m/s-mal megy, tehát ez nem szűk).

---

## 9. Jutalmak és büntetések, és hogyan számolódnak

### Hogyan adódnak össze

- **Minden lépésben**, minden robotra külön: `r_t = Σᵢ wᵢ · fᵢ(állapot) · dt`, ahol dt = 7/360 s. Az Isaac Lab reward
  manager minden tagot a lépésidővel szoroz, így a súly „jutalom per másodperc”, egy teljes epizódra pedig
  `wᵢ · átlag(fᵢ) · T`. A 0 súlyú tagokat nem számolja.
- A PPO **nem epizódösszeget** használ, hanem lépésenkénti **diszkontált hozamot**:
  `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …`, γ = 0,99 lépésenként = **0,596 másodpercenként**. (A γ a
  diszkontfaktor: egy k lépéssel későbbi jutalom γᵏ-szorosan számít.)
- A wandb `Mean reward` a befejezett epizódok diszkontálatlan összege; az `Episode_Reward/<tag>` az epizódösszeg
  osztva az epizódhosszal (másodpercenkénti átlag).

### A jutalomtagok (jelenlegi súlyokkal, 9 s)

| Tag | Képlet (lépésenként, dt-vel szorozva) | Súly | Forrás |
|---|---|---|---|
| `final_position` (fő jutalom) | `1/T_r · 1/(1 + d²)`, **csak az utolsó T_r = 2 s-ban**; d = cél-távolság [m] | +10 | Rudin (1) |
| `exploration_bias` | `cos(v, cél iránya)` = a sebesség és a célirány szögének koszinusza; automatikusan kikapcsol | +0,5 | Rudin (3) |
| `wheel_torque_l2` | `Σ τ²` a 4 kerékre [N²m²] | −0,00312 | Rudin (joint torque); Xie et al. 2020: elektromos veszteség |
| `action_rate_l2` | `‖aₜ − aₜ₋₁‖²` (nyers akciók) | −18 | Rudin (action rate) |
| `wheel_slip_l2` | `Σ (R·ω − gördülési sebesség az IK szerint)²` [m²/s²] | −6,944 | a láb-gyorsulás kerekes megfelelője; Xie et al.: súrlódási veszteség |
| `wheel_acc_l2`, `stalling`, `action_l2`, `obstacle_proximity`, `collision`, `ang_vel_xy_l2` | | 0 (kikapcsolva) | |

**A fő jutalom:** ha a robot az utolsó 2 s-ban végig a célon áll, az epizódösszeg `10 · (1/2 s) · 2 s = 10`. Ez a
teljes jutalomskála viszonyítási alapja. Mivel csak a végén jár, a policy szabadon választhat utat és sebességet. A
`1/(1+d²)` 1 m-en 0,5-öt, 0,5 m-en 0,8-at ad — a cél közvetlen közelében kicsi a „húzóerő”.

**Exploration bias:** a tanítás elején szinte soha nem ér célba a robot, így a fő jutalom nem ad gradienst; a bias a
cél felé mutató sebességet jutalmazza. Kikapcsol, ha a fő jutalom (súlyozatlan) epizódösszegének mozgóátlaga
(`0,99·régi + 0,01·új` a befejezett epizódokra) eléri a maximum **50%-át**. A pillaros runokban ez a ~130.
iterációnál történt.

**Büntetés-súlyok kalibrálása:** `w = −f · 10 / (m_ref · T)`, ahol `m_ref` a büntetett mennyiség lépésenkénti
átlaga a büntetés nélküli referencia-policynél. Így a büntetés a referencia viselkedésnél a fő jutalom maximumának
`f`-részét vonná le. Az elfogadott értékek: nyomaték f = 0,2 (m_ref = 91,6), akcióváltás f = 0,05, slip f = 0,1
(m_ref = 0,016, T = 9 s).

### Nagyságrendek (a jelenlegi `slip_f0.1` policy mért mozgásával, 9 s)

| Tag | Epizódonként | A fő jutalom maximumához (= 1) | Az epizód elejéről, diszkontálva |
|---|---|---|---|
| `final_position` (max.) | +10 | 1 | 0,168 (!) |
| `wheel_torque_l2` | −0,047 | −0,5% | a diszkontált fő jutalom −6% |
| `action_rate_l2` | −0,056 | −0,6% | −7% |
| `wheel_slip_l2` | −0,174 | −1,7% | −22% |
| `exploration_bias` (max., amíg aktív) | +4,5 | 0,45 | |

A kész policynél a büntetések epizódösszege kicsi, mert a policy megtanulta elkerülni őket (a nyomaték² 91,6-ról
1,7-re esett). A diszkont miatt azonban az epizód elején a büntetések sokkal erősebbek: a 9 s múlva járó fő jutalom
jelenértéke csak 0,168, a büntetések viszont azonnal járnak. A tanítás elején (véletlen policy) a büntetések a fő
jutalom maximumának többszörösei (akcióváltás ~170%, slip ~470%) — ezért omlik össze gyorsan az akció szórása.

---

## 10. Epizód vége (termináció)

| Feltétel | Mikor | Következmény |
|---|---|---|
| ütközés | a lidar szerint egy oszlop 3 cm-nél közelebb van a robot 0,66 × 0,60 m-es téglalapjához | az epizód azonnal véget ér → a még meg nem kapott fő jutalom elvész (ez az implicit ütközés-büntetés) |
| felborulás | dőlés > 60° | mint az ütközés |
| idő lejárt | 9 s | **nem truncation**: a hátralévő idő megfigyelés, így a critic tudja, hogy utána nincs több jutalom (nincs bootstrap), ahogy a cikkben |

Az ütközést azért a lidarból számoljuk és téglalappal, mert a korábbi kör alakú lábnyom miatt az ütközések 90%-a hamis
volt (az oszlop még > 10 cm-re volt a robot oldalától).

---

## 11. Curriculum

`terrain_levels_goal` ([`curriculums.py`](../source/mecanum_ws/mecanum_ws/tasks/manager_based/mecanum_navigation/mdp/curriculums.py)),
minden epizód végén, robotonként:

- **fel** egy szinttel, ha az epizód végén 0,5 m-en belül volt a célon;
- **le** egy szinttel, ha ütközött/felborult, vagy 2 m-nél messzebb maradt;
- különben marad.

A robotok a 0–2. szinten indulnak (`max_init_terrain_level = 2`). Az Isaac Lab terep-importere a legfelső szintet
teljesítő robotot egy véletlen szintre küldi. A tanítás végére az átlagos szint ~5,5–6,4.

**Mellékhatás — felejtés:** a curriculum a tanítási adatot a nehéz szintek felé tolja. A 7 s-os run 400 iterációs
folytatásánál a policy a nehéz szinteken kicsit javult, de a könnyű szinteken (0–1.) és oszlop nélküli terepen
nagyot romlott (sík terepen 93% → 50%): „elfelejtette” az üres környezetet. A tanítási átlagjutalom ezt nem mutatta.
Egy checkpoint mentésekor a curriculum-állapot nem mentődik, folytatáskor 0–2. szintről indul újra.

---

## 12. A neurális háló

RSL-RL `ActorCritic`, külön actor és critic hálóval, bemenet-normalizálással:

```
actor:  89 → 256 → 128 → 64 → 4   (ELU aktiváció)   64 452 paraméter + 4 szórás-paraméter
critic: 89 → 256 → 128 → 64 → 1   (ELU aktiváció)   64 257 paraméter
```

- Az actor egy **Gauss-eloszlás** várható értékét adja ki; a szórás állapotfüggetlen, tanulható paraméter
  (kezdetben 0,5). Tanításkor ebből mintavételez (felfedezés), kiértékeléskor a várható értéket használja
  (determinisztikus).
- A critic a várható diszkontált hozamot (állapotértéket) becsli; csak a tanításhoz kell.

---

## 13. PPO: konfiguráció és egy iteráció menete

[`rsl_rl_ppo_cfg.py`](../source/mecanum_ws/mecanum_ws/tasks/manager_based/mecanum_navigation/agents/rsl_rl_ppo_cfg.py):

| Paraméter | Érték |
|---|---|
| lépés / env / iteráció | 48 (≈ 0,93 s szimulált idő), mint Rudinnál |
| envek | 3072 → 147 456 minta iterációnként |
| epoch / minibatch | 5 / 4 → 20 gradiens-lépés iterációnként (minibatch: 36 864 minta) |
| tanulási ráta | 1e-3, **adaptív**: ha a KL > 2 × 0,01, a ráta ÷1,5 (min. 1e-5); ha < 0,01 / 2, ×1,5 (max. 1e-2) |
| clip | 0,2 |
| γ (diszkont) / λ (GAE) | 0,99 / 0,95 |
| value loss | súly 1,0, clipped |
| entropy-bónusz | **0** (eredetileg 0,005) |
| max. gradiens-norma | 1,0 |
| iterációk | pillaros runok: 400 (+400 folytatás) |

**Egy iteráció:**
1. **Rollout:** mind a 3072 robot 48 lépést tesz a jelenlegi policyvel (mintavételezett akciókkal); egy 9 s-os
   epizód (463 lépés) így ~10 iteráción ível át, a robotok egymástól függetlenül resetelnek.
2. **Előnyök (GAE):** a lépésenkénti jutalmakból és a critic becsléséből. A 48 lépéses ablak végén a folytatást a
   critic becsli.
3. **Frissítés:** 5 epoch × 4 minibatch; a clipped PPO-célfüggvény, a value loss és (itt nulla) entrópia.

**Miért entropy = 0?** Csak pozitív jutalommal az entrópia-bónusz miatt a szórás 0,5-ről 4,3-ra nőtt (a vágott
akcióknál a szélesebb eloszlás szinte semmibe sem kerül), a robotok rángatóztak. **Az érem másik oldala:** az
akcióváltás-büntetés közvetlenül bünteti a zajt, így a szórás a tanítás során 0,5-ről ~0,002-re omlik, a felfedezés
megszűnik, és az adaptív ütemező a tanulási rátát a minimumra viszi — a pillaros runok task rewardja a ~125.
iteráció után platón van.

---

## 14. A tanítás története

Részletesen, dátummal és wandb-linkkel: [`EXPERIMENTS.md`](../EXPERIMENTS.md).

| Időszak | Mit csináltunk | Eredmény / tanulság |
|---|---|---|
| 09-25 | robot generálása, validáció | 14 görgő (64-link korlát), lengő tengely, rugalmas kontaktus, motor-armatúra |
| 09-26 | Rudin-jutalom + minden büntetés, oszlopok, 1536 env | 900. iteráció: 65% siker, 22% ütközés — kiderült, hogy az ütközések 90%-a hamis volt (kör alakú lábnyom) → téglalap |
| 09-26/27 | súly-hangolás (`tune_rewards.py`), korai leállítással | a görbék ~200 iteráció után ellaposodtak, ~900 után sok ütközés; a robotok lassan mentek (gyanú: a büntetéseket minimalizálják) → átállás csak pozitív jutalomra |
| 09-27 | csak pozitív jutalom, oszlopok | 72,8% siker, 24,8% ütközés; entrópia-robbanás (std 0,5 → 4,3) → `entropy_coef = 0` |
| 09-28 | sík terep, csak pozitív jutalom | **100% siker**; a büntetések fokozatos visszakapcsolása (warm start) alig hatott — a súlyok túl gyengék voltak |
| 09-29 | büntetések egyenként, nulláról, kalibrált súllyal, 7 s (`penalty_study.py`) | elfogadva: nyomaték −0,00312 (nyomaték² −97%), akcióváltás −18 (−85%); kerékgyorsulás mind a 4 próbán elbukott |
| 09-29 | vissza az oszlopokra ugyanezekkel a súlyokkal | a videón a robotok álltak → **PhysX patch-buffer overflow** (lásd lent) → javítva |
| 09-29 | célreward súlya (`goal_weight_sweep.py`) | 10-es súllyal már minden szinten elindul a cél felé: 41,9% siker, 14,5% ütközés (`goalw_10`) |
| 09-29/30 | `goalw_10` folytatása +400 iterációval | rosszabb (38,7%): felejtés a curriculum miatt + nincs felfedezés |
| 09-30 | 9 s + slip-büntetés (`slip_sweep.py`), nulláról | f = 0,2 túl erős (30,3%); **f = 0,1 (w = −6,944) elfogadva: 52,8% siker**, 16,4% ütközés, slip² −90% |
| 09-30 | a `slip_f0.1` folytatása +400 iterációval | fut |

**A jelenleg legjobb pillaros policy** (`2026-09-30_03-27-35_pillars9s_slipw_slip_f0.1/model_399.pt`,
2048 epizód, cél ≥ 1 m):

| | Érték |
|---|---|
| siker (0,5 m-en belül a végén, ütközés nélkül) | 52,8% |
| siker a 0–3. / 4–6. / 7–9. szinten | 74% / 47% / 30% |
| ütközés | 16,4% |
| medián odaérés | 5,5 s (a 9 s-ból) |
| mozgás közbeni medián sebesség | 0,59 m/s (a lehetséges ~2 m/s-ból) |
| előre haladás (0–20°) / hátrafelé (135–180°) aránya | 59% / 17% |
| forgás menet közben / a célnál (medián) | 0,20 / 0,47 rad/s |

---

## 15. Hibák és tanulságok

1. **> 64 link a GPU-n:** figyelmeztetés nélkül felrobban sok env mellett → 14 görgő.
2. **Compliance csak a talaj anyagán hat** a PhysX-ben.
3. **Explicit motormodell stabilitása:** `d·dt/I < 2` kell → armatúra.
4. **Kör alakú lábnyom** az ütközéshez: 90% hamis ütközés → téglalap a lidarból.
5. **Entrópia-robbanás** csak pozitív jutalommal → entropy = 0; de ekkor a büntetések összeomlasztják a szórást.
6. **A büntetés „elfogadása” a hatás mérése nélkül félrevezető:** az első fokozatos visszakapcsolásnál minden lépés
   „átment”, de a mozgás alig változott → azóta a célzott metrikának legalább 20%-kal csökkennie kell.
7. **PhysX patch-buffer overflow:** a pillaros tanítás 3072 envvel az első lépéstől túlcsordult (~50 robot egy
   csempén), a PhysX kontaktokat dobott el, a robotok egy része elvesztette a tapadást és állt. A kevés envvel futó
   kiértékelésen ez nem látszott. Javítás: `gpu_max_rigid_patch_count = 2¹⁹`; azóta minden logban számoljuk.
8. **A curriculum felejtést okozhat,** és a tanítási átlagjutalom ezt elfedi → szintenként és sík terepen is
   ki kell értékelni.
9. **A folytatás (warm start) nem mindig javít:** ha a szórás már összeomlott, a policy csak elsodródik.

---

## 16. Jelenlegi állapot és javasolt következő lépések

**Fut:** a `slip_f0.1` policy folytatása a `model_399`-ből, ugyanazokkal a súlyokkal, +400 iteráció
(`pillars9s_slipf0.1_cont800`). A végén automatikus kiértékelés, mozgásdiagnosztika, `EXPERIMENTS.md`-sor és push.

**Javaslatok (fontossági sorrendben):**

1. **Testsebességes (IK) akciótér kipróbálása** ugyanezzel a jutalommal (`Mecanum-Navigation-Pillars-IK-v0`) — várhatóan
   kevesebb forgás és csúszás, gyorsabb tanulás; jó összehasonlítás a dolgozatba.
2. **Felfedezés megtartása:** alsó korlát a szórásra vagy kis entrópia-bónusz (pl. 0,001), mert a jelenlegi runok
   a ~125. iteráció után már nem tanulnak.
3. **Felejtés ellen:** a robotok egy része (pl. 20%) mindig véletlen szinten induljon.
4. **Checkpoint-választás:** több checkpointot kiértékelni (szintenként + sík terepen), és a legjobbat folytatni.
5. **Forgás a célnál:** ha megmarad, egy yaw-sebesség büntetés a célközelben (a slip-büntetés erre nem hat, mert a
   helyben forgás kinematikailag konzisztens, nem csúszás).
6. **Valódi robot felé:** súrlódás/csillapítás identifikálása, a Gazebo-val való keresztvalidáció.

---

## 17. Eszközök és fájlok

| Fájl | Mire való |
|---|---|
| `scripts/tools/generate_mecanum_robot.py` | a robot URDF generálása |
| `scripts/tools/validate_mecanum_wheels.py`, `summarize_validation.py` | fizikai validáció |
| `source/mecanum_ws/mecanum_ws/robots/mecanum.py` | robot-konstansok, motor, anyag |
| `.../mecanum_navigation/navigation_env_cfg.py` | a navigációs feladat (terep, megfigyelés, akció, jutalom, curriculum) |
| `.../mecanum_navigation/mdp/*.py` | saját jutalom-, megfigyelés-, cél-, curriculum- és terminációs függvények |
| `.../mecanum_navigation/agents/rsl_rl_ppo_cfg.py` | PPO és a háló |
| `scripts/rsl_rl/train.py`, `play.py` | tanítás, visszajátszás |
| `scripts/rsl_rl/evaluate_navigation.py` | determinisztikus kiértékelés szintenként (siker, ütközés iránya, sebességprofil, elindulás, mozgásminőség) |
| `scripts/tools/motion_diagnostics.py` | mozgás-statisztika és ábra (irány, forgás, kerékparancs-remegés, álló robotok) |
| `scripts/tools/penalty_study.py` | büntetések egyenként, kalibrált súllyal (sík terep) |
| `scripts/tools/goal_weight_sweep.py` | a célreward súlyának emelése |
| `scripts/tools/slip_sweep.py` | a slip-büntetés hangolása |
| `scripts/tools/train_eval_log.py` | egy tanítás felügyelet nélkül: tanítás (vagy folytatás) → kiértékelés → diagnosztika → `EXPERIMENTS.md` → push |
| `EXPERIMENTS.md` | minden tanítás eredménye dátummal és wandb-linkkel |
