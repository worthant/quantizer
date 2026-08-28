[![README in English](https://img.shields.io/badge/README-English-blue?style=flat-square)](README.md)

# atomic-quantizer

Конвейер квантизации, из которого вышли все GGUF- и MLX-релизы организации
[AtomicChat](https://huggingface.co/AtomicChat) на Hugging Face.

Два шелл-файла. Один копируется на арендованную машину, сорсится, дальше функции
вызываются по одной на переднем плане. Ничего не уходит в фон, ничего не
прячется, каждая функция говорит, что она делает, и громко останавливается,
когда чего-то не хватает.

| файл | что это |
| --- | --- |
| `foundry.sh` | GGUF-сторона. Сетап бокса, сборка llama.cpp, корпус, imatrix, лестница квантов, KLD, выгрузки |
| `foundry-mlx3.sh` | MLX-сторона. Самодостаточна, `foundry.sh` ей не нужен |
| `auto_fmt.py` | универсальный рендерер чата. Канонический экземпляр лежит в `calib-corpora/tools/` |
| `probe.sh` | не проверен и не документирован. Прочитать перед запуском |

Подробно по каждому файлу: [docs/scripts.md](docs/scripts.md).

## Установка на арендованный бокс

```bash
# 1. подключиться. TERM=xterm снимает рассинхрон terminfo внутри контейнера.
#    Шаблон инстанса сразу кидает в tmux.
vastai ssh-url <instance-id>
TERM=xterm ssh root@<ip> -p <port>

# 2. токен. ssh-логин не наследует окружение контейнера,
#    поэтому экспортируем руками один раз на бокс.
export HF_TOKEN=hf_...

# 3. забрать тулбокс и загрузить его
curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry.sh -o /foundry.sh
source /foundry.sh

# 4. чтобы новая вкладка tmux не была амнезийной
persist_shell
save_token

# 5. проверить, что из файла ничего не потерялось
selfcheck
```

```
all functions present, version 2026-08-21.01
```

> [!NOTE]
> `raw.githubusercontent.com` стоит за CDN и может отдать файл, которому
> несколько минут. Если только что запушенное не появляется, клонируйте, `reload`
> ходит через git ровно по этой причине:
>
> ```bash
> git clone https://github.com/AtomicBot-ai/atomic-quantizer /quantizer
> source /quantizer/foundry.sh
> ```

> [!WARNING]
> `save_token` пишет `HF_TOKEN` в `~/.bashrc` на боксе, именно это делает новую
> вкладку tmux рабочей. Бокс одноразовый, токен нет. Отзывайте его, когда работа
> закончена, и никогда не снапшотьте такой инстанс в образ.

## Четыре команды для ориентирования

```bash
status        # чеклист ЭТОГО бокса, у каждой невыполненной строки написана команда
plan          # что уже лежит в репозиториях на Hugging Face
help_me       # все функции, по группам
menu          # те же команды выбором по номеру
```

`status` используется постоянно и выглядит так:

```
  [x] preset: AtomicChat/Qwen3.8-27B-GGUF
  [x] cuda checked
  [x] tools installed
  [x] llama.cpp built
  [x] eval corpus
  [ ] original weights in /src  ->  run:  get_upstream
  [x] bf16 on disk
  [x] quants on disk: 14 files
  [ ] reference built        ->  run:  base
  [ ] kld logs: 0       ->  run:  kld_all
  [ ] bench logs: 0     ->  run:  bench_all
  [ ] results.json        ->  run:  results

  imatrix:
    [x] calibration corpus
    [x] IM_MODEL: Qwen3.8-27B-BF16.gguf
    [ ] no imatrix          ->  run:  im_plan N, then im_shard I N on each box

  full command list: help_me      pick by number: menu
  remote view: plan               integrity: selfcheck
```

У каждой невыполненной строки написана команда. Порядок держать в голове не
нужно.

## Подготовка машины

```bash
check              # диск, GPU, ядра, RAM
cuda_arch          # читает compute capability с карты
cuda_check         # тулкит против драйвера хоста, убирает битые compat-либы
setup              # тулчейн и python-пакеты, создаёт директории
build 120          # число из вывода cuda_arch
gpu_test           # в списке должны быть все GPU
```

`cuda_arch` печатает точную строку сборки, потому что число не угадывается по
названию карты (H100 и H200 обе 90, RTX 5090 это 120, B200 это 100):

```
compute capability 12.0 on NVIDIA GeForce RTX 5090
build with:  build 120
```

Две вещи, которые `setup` не ставит, а они нужны на каждом боксе:

```bash
apt-get install -y vim
fix_tmux           # прокрутка мышью и 200 тысяч строк истории
```

## Выбор модели

```bash
use_model qwen3.8-27b Qwen/Qwen3.8-27B
```

```
recipe name   : qwen3.8-27b
our gguf repo : AtomicChat/Qwen3.8-27B-GGUF
metrics repo  : AtomicChat/Qwen3.8-27B-GGUF-metrics  (type: dataset)
upstream      : Qwen/Qwen3.8-27B
eval corpus   : /eval/neutral.txt
context       : 4096

Run plan to see what exists where and what to do next.
```

Имена репозиториев выводятся из upstream-id, а не набираются руками.
Несуществующий на хабе id отвергается, поэтому опечатка не расползётся по всем
последующим шагам. Если id неизвестен точно: `find_repo qwen3.8`.

Если бокс до этого использовался под другую модель, `status` про это скажет, и
дальше:

```bash
clean_run          # сносит /logs /kld /imatrix, веса не трогает
clean_gguf         # предлагает удалить веса, принадлежащие другой модели
```

## Полный GGUF-релиз

Выполнять по одной команде, читая вывод каждой.

```bash
# --- веса ----------------------------------------------------------------
get_bf16                        # забрать наш опубликованный BF16-файл
# для модели, которую ещё никто не делал, вместо этого:
#   get_upstream ; setup_convert ; make_bf16 ; check_blocks

# --- калибровка ----------------------------------------------------------
ls_corpora                      # какие сборки есть в calib-corpora
get_calib qwen3.8-27b           # забрать builds/<name>/calib_train.txt
pick_model                      # интерактивно: выбрать BF16-файл для сбора
im_size                         # сколько это чанков

# --- importance matrix, один бокс ----------------------------------------
im_shard 0 1
# на шести боксах смотри раздел ниже

# --- эталон --------------------------------------------------------------
get_eval                        # eval/neutral
eval_size
base                            # пишет эталонные логиты, 4 до 8 минут
push_base

# --- лестница ------------------------------------------------------------
bits                            # из чего состоит модель, по ролям тензоров
write_ladder                    # пишет /ladder.txt
ladder /ladder.txt --dry        # только размеры, ничего не строится
ladder /ladder.txt              # собрать, замерить и опубликовать каждую ступень

# --- замеры --------------------------------------------------------------
use_gpus 0,1                    # держать прогоны квантов на сетапе, близком к читателю
kld_all
bench_all
results

# --- публикация ----------------------------------------------------------
audit                           # опубликовано но не замерено, и наоборот
push_quants
push_logs
push_results
```

`eval_size` и `results` это те два вывода, которые читаются внимательнее всего:

```
corpus : /eval/neutral.txt
size   : 1414908 bytes, roughly 353727 tokens
at ctx 4096 that is about 86 chunks
```

```
file                                               GB   mean KLD    top-1     pp512     tg128
Qwen3.8-27B-AD-IQ2_XS                            8.71   0.201443    89.12     1840.2      41.7
Qwen3.8-27B-AD-IQ3_XXS                          10.44   0.118207    91.88     1795.6      38.9
Qwen3.8-27B-AD-IQ4_XS                           16.81   0.015799    97.31     1702.4      31.2
...
written to /logs/results.json
```

`kld_all` пропускает уже замеренное, поэтому он же является командой догона,
если бокс собрал кванты раньше, чем появился его эталон. `KLD_FORCE=1 kld_all`
переделывает всё заново.

Чужие сборки меряются так же, против нашего эталона:

```bash
get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'
kld_ext
```

### Абляция: выбор раскладки, а не ступени

`ladder` решает, какие размерные классы публикуются. `ablate` решает, какую
раскладку использует ступень, и это другой вопрос: каждый кандидат строится в
одном размере, меряется против одного эталона и после этого удаляется.

```bash
write_ablation                  # пишет /ablate.txt, восемь кандидатов
ablate --dry                    # только предсказанные размеры, проверить что они в одной полосе
ablate                          # собрать, замерить, удалить, дальше
ablate_report                   # таблица, читается уже после удаления файлов
```

```
candidate                GB    mean KLD    vs base      median        99%    top-1
A-uniform             16.81    0.015799         --    0.002100    0.31000    97.31
B-edge4               17.08    0.014492     -8.3 %    0.001980    0.29500    97.44
C-edge16              17.82    0.009811    -37.9 %    0.001510    0.24100    97.98
```

Размеры, выпавшие за полосу в 2 процента вокруг базового, помечаются: кандидат,
который больше и лучше, может говорить только о том, что биты помогают.
`ABL_KEEP=1` оставляет файлы, `ABL_FORCE=1` перемеряет.

> [!WARNING]
> Диапазоны блоков в `write_ablation` рассчитаны на модель из шестидесяти
> четырёх блоков. На другой глубине они дадут рабочий файл, который поднимает не
> те блоки, а это самый трудно замечаемый вид ошибки. Сверьте их по `bits`.

### Против комьюнити, в их размерах

```bash
get_external unsloth/Qwen3.8-27B-GGUF '*UD-IQ3_XXS*'
kld_ext
results
vs_community
```

```
their build                                      GB   their KLD  ours there     we are    top-1
unsloth--Qwen3.8-27B-UD-IQ3_XXS               13.70    0.154161    0.042203    +72.6 %    89.90
bartowski--Qwen3.8-27B-IQ4_XS                 16.05    0.053775    0.020087    +62.6 %    95.10
```

Размеры никогда не совпадают, поэтому здесь наша лестница интерполируется в их
точный размер, а не сравнивается наша сборка на 17.6 ГБ с их на 16.1. Для
сборок, выпавших за края лестницы, честного сравнения нет, и это написано прямо,
вместо экстраполяции.

## Importance matrix на N боксах

Тот самый трюк, который превращает шестичасовой imatrix в получасовой, и он
точный, а не приближённый: матрица это сумма квадратов активаций, а сумма
делится. Каждый узел читает **один и тот же файл корпуса** и берёт свой кусок по
диапазону чанков, поэтому объединение побайтово совпадает с тем, что дал бы один
узел.

На каждом боксе: установка, `use_model`, `get_calib`, `get_bf16`, `pick_model`.

На одном боксе:

```bash
im_plan 6
```

```
6 nodes, about 1620 chunks each
run get_calib and set IM_MODEL to the same file on every box, then:

  node 0:   im_shard 0 6
  node 1:   im_shard 1 6
  node 2:   im_shard 2 6
  node 3:   im_shard 3 6
  node 4:   im_shard 4 6
  node 5:   im_shard 5 6

then on any one box:   im_merge 6
```

По одной строке в каждый бокс. Каждый шард сам выгружается в репозиторий метрик,
когда досчитывает, поэтому файлы руками никто не перекладывает. Дальше на любом
боксе:

```bash
im_status                       # что здесь, что в репозитории, какие диапазоны покрыты
im_merge_all
```

Арендованные машины никогда не одинаковы, поэтому если один узел медленнее
остальных, его диапазон режется и раздаётся тем, кто освободился:

```bash
im_range 7275 700 5a            # чанки с 7275 по 7974
im_range 7975 0   5b            # с 7975 до конца корпуса
IM_SKIP="shard-5-of-6" im_merge_all
```

Подробно: [docs/imatrix-sharding.md](docs/imatrix-sharding.md).

## Полный MLX-релиз

`foundry-mlx3.sh` самодостаточна и не собирает llama.cpp, что экономит пятнадцать
минут на боксе, который занят только MLX.

> [!IMPORTANT]
> Её умолчания указывают на Qwen3.8-27B и читаются в момент сорса файла.
> Для любой другой модели сначала экспорт, потом source:
>
> ```bash
> export MLX3_UP=Qwen/Qwen3.8-27B
> export MLX3_STEM=Qwen3.8-27B
> export MLX3_REF=/mlx/$MLX3_STEM-MLX-bf16
> export MLX3_TEACHER=/mlx/$MLX3_STEM-MLX-8bit
> export MLX3_METRICS=AtomicChat/$MLX3_STEM-MLX-metrics
> ```

```bash
curl -sL https://raw.githubusercontent.com/AtomicBot-ai/atomic-quantizer/main/foundry-mlx3.sh -o /mlx3.sh
source /mlx3.sh

mlx3_setup
mlx3_check                      # ЧИТАТЬ. Если написано "cpu", ниже ничего не работает
mlx3_bench                      # десятки TFLOP/s вместо сотен = тензорные ядра не задействованы
mlx3_persist

mlx3_get ref                    # bf16-чекпоинт, точка отсчёта. Около 51 ГБ
mlx3_get eval
mlx3_cache                      # один эталонный проход, переиспользуется всеми замерами

mlx3_get src
mlx3_quant 4 64                 # обычная ступень
mlx3_clip /mlx/Qwen3.8-27B-MLX-4bit    # лучшее округление, размер совпадает побайтово
mlx3_kld /mlx/Qwen3.8-27B-MLX-4bit-CLIP
mlx3_table
mlx3_push /mlx/Qwen3.8-27B-MLX-4bit-CLIP 4bit-CLIP
```

`mlx3_table` ставит твои сборки рядом с цифрами, которые надо побить:

```
build                                            GB    mean KLD   top-1 %       ppl
Qwen3.8-27B-MLX-3bit-CLIP                     12.70    0.180912     92.44    4.9871
Qwen3.8-27B-MLX-4bit-CLIP                     16.05    0.051004     97.02    4.5983

targets, and the best of ours at or below each size:
  13.70 GB  0.154161  maglun 3.80bpw      ours 0.180912 at 12.70 GB -> behind by 17.4 %
  16.05 GB  0.053775  WaveCut 4bit-DWQ    ours 0.051004 at 16.05 GB -> BEATEN by 5.2 %
```

Ещё два рычага, оба описаны в [docs/runbook-mlx.md](docs/runbook-mlx.md):
`mlx3_gen` и `mlx3_build` для потензорной раскладки бит, которая стоит около
двадцати процентов там, где клиппинг стоит два-пять, и `mlx3_dwq` для
дистилляции, которая окупается между 2 и 4 битами.

`mlx3_box hub` и `mlx3_box dwq` печатают точный список команд для плана на два
бокса, с контрольной точкой на каждом этапе.

## Когда что-то сломалось

```bash
lastlog                         # открыть свежий лог в пейджере с поиском
lastlog kld                     # или свежий лог по шаблону
reload                          # подтянуть текущий файл из git и пересорсить
selfcheck                       # проверить, что ни одна функция не потерялась
```

`reload` печатает коммит, на который встал. Он ходит через git, а не по прямой
ссылке на файл, потому что оба http-пути к GitHub кэшируются и могут отдать
копию, которой несколько минут:

```
commit: 4f2a1c9 2026-08-21 fix: pin the MTP block before low bit rungs
here: 2026-08-21.01    at that commit: 2026-08-21.02
now running 2026-08-21.02
```

Всё, что хоть раз стоило часов, записано в
[docs/troubleshooting.md](docs/troubleshooting.md). Читать до того, как начинать
что-то отлаживать.

## Документы

| документ | когда читать |
| --- | --- |
| [docs/pipeline-map.md](docs/pipeline-map.md) | первым. Что в каком репозитории и в какой директории |
| [docs/runbook-gguf.md](docs/runbook-gguf.md) | полный GGUF-релиз с объяснением каждого шага |
| [docs/runbook-mlx.md](docs/runbook-mlx.md) | полный MLX-релиз |
| [docs/imatrix-sharding.md](docs/imatrix-sharding.md) | схема шардирования подробно |
| [docs/renting-boxes.md](docs/renting-boxes.md) | подбор железа и на что уходят деньги |
| [docs/troubleshooting.md](docs/troubleshooting.md) | поломки, которые в первый раз стоили часов |
| [docs/security-audit.md](docs/security-audit.md) | доступы, что уезжает с бокса, что поменять |
| [docs/rough-edges.md](docs/rough-edges.md) | известные баги и незакрытые места |
| [docs/scripts.md](docs/scripts.md) | что представляет собой каждый файл в корне |

## Правило, на котором держатся все цифры

Значение KL-дивергенции что-то значит только относительно эталона, против
которого оно снято. Цифры, снятые против разных эталонов, на разных корпусах или
при разной длине контекста, в одну таблицу не ставятся. Все опубликованные
сравнения в этом конвейере получены так: файлы другого паблишера скачиваются и
перемеряются против нашего эталона, а не берутся из его публикации.
