# Kísérleti napló

Minden tanítás egy sorban: mi változott, melyik kódállapoton (commit), mi lett az eredmény, hol van a wandb-run.
A "siker" mindig a determinisztikus policy kiértékelése (`scripts/rsl_rl/evaluate_navigation.py`): a robot az
epizód végén 0,5 m-en belül van a célon, ütközés nélkül. Wandb-projekt:
<https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation>

A git-követés 2026-09-28-án indult (`243ac0c`); a korábbi runok kódja nagyjából ennek az állapotnak felel meg.

## 1. Oszlopos terep, a cikk szerinti jutalom + büntetések

| Dátum | Run (wandb) | Változás | Eredmény |
|---|---|---|---|
| 09-26 | [2026-09-26_13-23-10](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/a7yo8x1j) | Rudin et al. 2022 jutalom (final position + exploration bias + stalling) + büntetések, 1536 env | 922. iterációnál leállt (SSH-bontás). `model_900`: siker 65,1%, ütközés 22,3%. **Hiba:** az ütközés köralakú lábnyommal számolt, az ütközések 90%-a hamis volt; téglalapra javítva ugyanez a policy 69,0% / 15,6%. |
| 09-26/27 | round1 study ([baseline](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/zx6u5l16), [gamma995](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/ucztqwgj), [speed](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/89feampn), [entropy](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/1sp1tp23)) | `scripts/tools/tune_rewards.py`, 3072 env, korai leállással | siker / ütközés: baseline 60,4% / 13,6%, gamma995 64,4% / 22,3%, **speed** (γ 0,995, 8 s epizód, könnyebb büntetések) 69,6% / 20,7%, entropy 67,3% / 15,7%. A `combo`, `precise`, `combo_gamma998` a `std` Hydra-kulcs hiánya miatt el sem indult (javítva). |

## 2. Csak pozitív jutalom (final position + exploration bias)

| Dátum | Run (wandb) | Változás | Eredmény |
|---|---|---|---|
| 09-27 | [current_weights](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/3xj1qo3v) | áttekintő kamera, jelenlegi súlyok | a 65. iterációnál leállítva (átállás csak pozitív jutalomra) |
| 09-27 | [task_and_bias_only](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/9zxirn4f) | oszlopos terep, minden büntetés kikapcsolva, 500 iteráció | siker 72,8%, ütközés 24,8% (korai, sarok-ütközések). **Hiba:** `entropy_coef = 0,005` mellett a szórás 0,5 → 4,3 (a tanítás közben a robotok rángatóztak); a célban 3,4 rad/s-mal pörög. |
| 09-28 | [flat_task_and_bias](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/p398yy43) (`868a2df`) | sík terep oszlopok nélkül, `entropy_coef = 0` | siker 100%, a célnál 3,6 cm, szórás 0,11, pörgés a célban 0,43 rad/s |

## 3. Büntetések lépésenkénti visszakapcsolása (sík terep, `96324d8`)

`scripts/tools/staged_penalties.py --study flat_penalties`; minden lépés az előző policyből folytat 150 iterációval.
Riport: `logs/staged/flat_penalties/report.md`.

| Lépés | Run (wandb) | Bekapcsolt büntetés | Siker | Célzott metrika (előtte → utána) |
|---|---|---|---|---|
| 0 | flat_task_and_bias | – | 100% | – |
| 1 | [torque](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/dy36e925) | `wheel_torque_l2 = −8e-4` | 100% | nyomaték²: 70,1 → 62,4 (−11%) |
| 2 | [action rate](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/ldd2jt1v) | `action_rate_l2 = −0,03` | 100% | akcióváltás²: 0,484 → 0,457 (−6%) |
| 3 | [acc](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/ebpd6jri) | `wheel_acc_l2 = −4e-6` | 100% | gyorsulás²: 5,98e4 → 5,74e4 (−4%) |
| 4 | [stalling](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/b0fsfdfw) | `stalling = −1` | 100% | álldogálás: 0,4% → 0,4% |
| 5 | [slip](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/qi2wo6xn) | `wheel_slip_l2 = −1` | 100% | csúszás²: 0,459 → 0,443 (−3%) |
| 6 | [action](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/1kq0btmq) | `action_l2 = −0,2` | 100% | akció²: 1,30 → 1,34 (+3%) |

**Következtetés:** minden lépés "átment", de a büntetések alig változtattak a mozgáson. Az elfogadási szabály csak
azt nézte, hogy a siker ne romoljon, azt nem, hogy a büntetés tényleg hat-e; a finomhangolt policy szórása pedig
már 0,11 volt, így 150 iteráció alatt csak keveset tudott változni.

A végső policy mozgása (`scripts/tools/motion_diagnostics.py`, 64 epizód, cél ≥ 2 m):
- gyors indulás (3 kerék 20 rad/s-on telítve, ~1,7 m/s), utána a cél előtti utolsó ~0,6 m-en 6 s-ig "kúszik";
  medián odaérés 5,6 s 3,75 m-re, pedig a jutalom csak a 10–12. s-ban jár;
- az idő 39%-ában 20–45°-ban ferdén, 16%-ában hátrafelé-ferdén halad, csak 21%-ában előre; az út 14%-kal hosszabb
  az egyenesnél, néha nagy kerülővel;
- a célban forog (0,5 rad/s), a kerekek átlagosan 2,3 rad/s-mal forognak, miközben a robot áll (csúszás);
- a kerékparancsok remegnek: kerekenként ~24 irányváltás másodpercenként.

## Büntetések egyenként, nulláról tanítva, 7 s epizód (study `flat7s`)

`scripts/tools/penalty_study.py --study flat7s`: minden policy nulláról, 400 iteráció, 3072 env.
Súly: `w = −f · 10 / (m_ref · 7 s)`, ahol `m_ref` a büntetett mennyiség lépésenkénti
átlaga az előző elfogadott policynél (így a büntetés a fő jutalom `f`-szeresét vonná le). Elfogadás: a siker / ütközés /
odaérés nem romlik, és a célzott metrika legalább 20%-kal csökken.

| Dátum | Run (wandb) | Lépés / próba | Új büntetés, súly (f) | Siker | Ütközés | Odaérés [s] | Célzott metrika: ref → új | Döntés |
|---|---|---|---|---|---|---|---|---|
| 2026-09-29 01:45 | [stage0_base](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/qrdfm9um) | 0 / 0 | – (csak pozitív jutalom) | 100.0% | 0.0% | 3.0 | – | accepted (gate) |
| 2026-09-29 03:02 | [stage1_wheel_torque_l2_a0](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/asvukm8q) | 1 / 0 | `wheel_torque_l2 = -0.00312` (f = 0.2) | 100.0% | 0.0% | 4.0 | torque2: 91.61 → 2.739 (-97%) | accepted |
| 2026-09-29 04:20 | [stage2_action_rate_l2_a0](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/p0y5jkxa) | 2 / 0 | `action_rate_l2 = -72.1` (f = 0.2) | 39.4% | 0.0% | 5.9 | action_rate2: 0.003965 → 0.0002689 (-93%) | rejected: too strong (success / arrival worse) -> weaker |
| 2026-09-29 05:37 | [stage2_action_rate_l2_a1](https://wandb.ai/varadipeter05-budapesti-m-szaki-s-gazdas-gtudom-nyi-egyetem/mecanum-navigation/runs/huaz2mef) | 2 / 1 | `action_rate_l2 = -36` (f = 0.1) | 61.8% | 0.0% | 3.5 | action_rate2: 0.003965 → 0.0005238 (-87%) | rejected: too strong (success / arrival worse) -> weaker |
