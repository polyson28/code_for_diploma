```
from tqdm import tqdm
import optuna
import torch.optim as optim
from optuna.trial import TrialState
from optuna.samplers import TPESampler, BaseSampler
from optuna.pruners import MedianPruner
from model import define_model, objective
```

```python
sampler = TPESampler(
    n_startup_trials=20,
    seed=42
)
pruner = MedianPruner(
    n_startup_trials=20,
    n_warmup_steps=5,
    interval_steps=3
)
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    pruner=pruner
)

# Ensure that we get 60 successful trials
n_successful_trials = 60
successful_trials = 0

pbar = tqdm(total=n_successful_trials, desc="Successful Trials", unit="trial")

while successful_trials < n_successful_trials:
    study.optimize(objective, n_trials=1, n_jobs=-1)
    trial = study.trials[-1]

    if trial.state == TrialState.COMPLETE:
        successful_trials += 1
        pbar.update(1)  # Update progress bar

# Close the progress bar
pbar.close()

# Print the best trial
best_trial = study.best_trial
print(f'Best trial: {best_trial.number}')
print(f'  Value: {best_trial.value}')
print(f'  Params: ')
for key, value in best_trial.params.items():
    print(f'    {key}: {value}')

# Print the number of completed trials
print(f'Number of successful trials: {successful_trials}')

fig = optuna.visualization.plot_optimization_history(study)
fig.show()
```
```
[I 2024-06-04 14:46:15,727] A new study created in memory with name: no-name-80b637a9-3fb8-4c0a-ab11-1198d169a47f
Successful Trials:   0%|          | 0/60 [00:00<?, ?trial/s][I 2024-06-04 14:49:12,239] Trial 0 finished with value: 0.00019901175269750382 and parameters: {'hidden_size': 164, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.09505278068571384, 'optimizer': 'RMSprop', 'lr': 0.009696654549413403, 'batch_size': 512}. Best is trial 0 with value: 0.00019901175269750382.
Successful Trials:   2%|▏         | 1/60 [02:56<2:53:34, 176.51s/trial][I 2024-06-04 14:51:54,856] Trial 1 finished with value: 0.0001747436396708137 and parameters: {'hidden_size': 381, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.24205810790006804, 'optimizer': 'Adam', 'lr': 0.0025222111556541275, 'batch_size': 1024}. Best is trial 1 with value: 0.0001747436396708137.
Successful Trials:   3%|▎         | 2/60 [05:39<2:42:43, 168.34s/trial][I 2024-06-04 14:54:33,240] Trial 2 finished with value: 0.0002613632738330682 and parameters: {'hidden_size': 264, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.28900715815758754, 'optimizer': 'RMSprop', 'lr': 0.005915710836231594, 'batch_size': 1024}. Best is trial 1 with value: 0.0001747436396708137.
Successful Trials:   5%|▌         | 3/60 [08:17<2:35:36, 163.79s/trial][I 2024-06-04 14:58:17,493] Trial 3 finished with value: 0.0001530145578082338 and parameters: {'hidden_size': 260, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.11820020543390096, 'optimizer': 'RMSprop', 'lr': 0.0013300137958104814, 'batch_size': 256}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:   7%|▋         | 4/60 [12:01<2:55:09, 187.66s/trial][I 2024-06-04 15:00:57,397] Trial 4 finished with value: 0.00030056556891079763 and parameters: {'hidden_size': 250, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.20049293481882874, 'optimizer': 'RMSprop', 'lr': 0.005479917288747638, 'batch_size': 2048}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:   8%|▊         | 5/60 [14:41<2:42:50, 177.65s/trial][I 2024-06-04 15:03:28,812] Trial 5 finished with value: 0.018525956218697074 and parameters: {'hidden_size': 303, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.21851968436642405, 'optimizer': 'SGD', 'lr': 0.0035467858255477567, 'batch_size': 1024}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  10%|█         | 6/60 [17:13<2:31:51, 168.73s/trial][I 2024-06-04 15:05:57,816] Trial 6 finished with value: 0.00022619908277007124 and parameters: {'hidden_size': 232, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.198718256394529, 'optimizer': 'Adam', 'lr': 0.005890386477995632, 'batch_size': 2048}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  12%|█▏        | 7/60 [19:42<2:23:20, 162.28s/trial][I 2024-06-04 15:09:18,973] Trial 7 finished with value: 0.0023950788723279557 and parameters: {'hidden_size': 322, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.17135806628647893, 'optimizer': 'SGD', 'lr': 0.0087048797347659, 'batch_size': 256}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  13%|█▎        | 8/60 [23:03<2:31:22, 174.66s/trial][I 2024-06-04 15:11:53,005] Trial 8 finished with value: 0.0002885752793104412 and parameters: {'hidden_size': 260, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.19855188044230976, 'optimizer': 'RMSprop', 'lr': 0.0015504531324069804, 'batch_size': 512}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  15%|█▌        | 9/60 [25:37<2:22:58, 168.21s/trial][I 2024-06-04 15:14:15,560] Trial 9 finished with value: 0.001440760514221011 and parameters: {'hidden_size': 100, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.07250465219775103, 'optimizer': 'RMSprop', 'lr': 0.007070123518324162, 'batch_size': 2048}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  17%|█▋        | 10/60 [27:59<2:13:34, 160.29s/trial][I 2024-06-04 15:16:42,047] Trial 10 finished with value: 0.014839503332278783 and parameters: {'hidden_size': 259, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.10138025568420748, 'optimizer': 'SGD', 'lr': 0.007754093137719924, 'batch_size': 2048}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  18%|█▊        | 11/60 [30:26<2:07:27, 156.07s/trial][I 2024-06-04 15:19:14,614] Trial 11 finished with value: 0.000632562110524527 and parameters: {'hidden_size': 372, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.06579460599782032, 'optimizer': 'RMSprop', 'lr': 0.0012854990380741502, 'batch_size': 2048}. Best is trial 3 with value: 0.0001530145578082338.
Successful Trials:  20%|██        | 12/60 [32:58<2:04:00, 155.00s/trial][I 2024-06-04 15:22:50,664] Trial 12 finished with value: 8.661206362571497e-05 and parameters: {'hidden_size': 219, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.09223952881108534, 'optimizer': 'Adam', 'lr': 0.0019015817916292032, 'batch_size': 256}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  22%|██▏       | 13/60 [36:34<2:15:54, 173.50s/trial][I 2024-06-04 15:25:06,639] Trial 13 finished with value: 0.008241050160247643 and parameters: {'hidden_size': 228, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.16270651031674443, 'optimizer': 'SGD', 'lr': 0.005814586803205671, 'batch_size': 1024}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  23%|██▎       | 14/60 [38:50<2:04:19, 162.16s/trial][I 2024-06-04 15:27:22,701] Trial 14 finished with value: 0.0035131051048517053 and parameters: {'hidden_size': 171, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.10005759994568068, 'optimizer': 'SGD', 'lr': 0.009681500442121497, 'batch_size': 1024}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  25%|██▌       | 15/60 [41:06<1:55:43, 154.29s/trial][I 2024-06-04 15:30:03,580] Trial 15 finished with value: 0.00010003296834173436 and parameters: {'hidden_size': 146, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.10721466218399732, 'optimizer': 'Adam', 'lr': 0.00914473360286396, 'batch_size': 512}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  27%|██▋       | 16/60 [43:47<1:54:36, 156.28s/trial][I 2024-06-04 15:32:19,707] Trial 16 finished with value: 0.01898461324813163 and parameters: {'hidden_size': 154, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.12328428484819358, 'optimizer': 'SGD', 'lr': 0.003199302286661457, 'batch_size': 1024}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  28%|██▊       | 17/60 [46:03<1:47:39, 150.22s/trial][I 2024-06-04 15:34:37,047] Trial 17 finished with value: 0.00028383253114655846 and parameters: {'hidden_size': 198, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.282955573978328, 'optimizer': 'RMSprop', 'lr': 0.005468515087973533, 'batch_size': 1024}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  30%|███       | 18/60 [48:21<1:42:26, 146.35s/trial][I 2024-06-04 15:37:08,088] Trial 18 finished with value: 0.0035652182326810513 and parameters: {'hidden_size': 265, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.08531511485532865, 'optimizer': 'SGD', 'lr': 0.007339932972422803, 'batch_size': 512}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  32%|███▏      | 19/60 [50:52<1:40:58, 147.76s/trial][I 2024-06-04 15:39:51,566] Trial 19 finished with value: 0.00023894003909047355 and parameters: {'hidden_size': 385, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.18932469538292152, 'optimizer': 'Adam', 'lr': 0.0008793984181408148, 'batch_size': 512}. Best is trial 12 with value: 8.661206362571497e-05.
Successful Trials:  33%|███▎      | 20/60 [53:35<1:41:39, 152.48s/trial][I 2024-06-04 15:43:26,803] Trial 20 finished with value: 7.88546025779464e-05 and parameters: {'hidden_size': 107, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.14545251134787668, 'optimizer': 'Adam', 'lr': 0.004380658470055198, 'batch_size': 256}. Best is trial 20 with value: 7.88546025779464e-05.
Successful Trials:  35%|███▌      | 21/60 [57:11<1:51:21, 171.32s/trial][I 2024-06-04 15:47:02,522] Trial 21 finished with value: 8.048975251788777e-05 and parameters: {'hidden_size': 96, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.14398188247641303, 'optimizer': 'Adam', 'lr': 0.004248327265683565, 'batch_size': 256}. Best is trial 20 with value: 7.88546025779464e-05.
Successful Trials:  37%|███▋      | 22/60 [1:00:46<1:56:56, 184.64s/trial][I 2024-06-04 15:50:38,358] Trial 22 finished with value: 7.761141363995969e-05 and parameters: {'hidden_size': 81, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.14227542626569964, 'optimizer': 'Adam', 'lr': 0.004104586122826222, 'batch_size': 256}. Best is trial 22 with value: 7.761141363995969e-05.
Successful Trials:  38%|███▊      | 23/60 [1:04:22<1:59:38, 194.00s/trial][I 2024-06-04 15:54:13,816] Trial 23 finished with value: 7.686285750012135e-05 and parameters: {'hidden_size': 92, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1419366445040498, 'optimizer': 'Adam', 'lr': 0.00414121933079195, 'batch_size': 256}. Best is trial 23 with value: 7.686285750012135e-05.
Successful Trials:  40%|████      | 24/60 [1:07:58<2:00:15, 200.44s/trial][I 2024-06-04 15:57:50,253] Trial 24 finished with value: 7.56167952990336e-05 and parameters: {'hidden_size': 81, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.14016515857923353, 'optimizer': 'Adam', 'lr': 0.004616113242823115, 'batch_size': 256}. Best is trial 24 with value: 7.56167952990336e-05.
Successful Trials:  42%|████▏     | 25/60 [1:11:34<1:59:43, 205.24s/trial][I 2024-06-04 16:01:29,032] Trial 25 finished with value: 7.676045626697107e-05 and parameters: {'hidden_size': 124, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.14173683596753966, 'optimizer': 'Adam', 'lr': 0.003788629220284353, 'batch_size': 256}. Best is trial 24 with value: 7.56167952990336e-05.
Successful Trials:  43%|████▎     | 26/60 [1:15:13<1:58:36, 209.30s/trial][I 2024-06-04 16:05:07,837] Trial 26 finished with value: 7.79833073237897e-05 and parameters: {'hidden_size': 123, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.16064861910523257, 'optimizer': 'Adam', 'lr': 0.004741998330659318, 'batch_size': 256}. Best is trial 24 with value: 7.56167952990336e-05.
Successful Trials:  45%|████▌     | 27/60 [1:18:52<1:56:41, 212.15s/trial][I 2024-06-04 16:08:46,346] Trial 27 finished with value: 7.80232880102663e-05 and parameters: {'hidden_size': 127, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1304895390350878, 'optimizer': 'Adam', 'lr': 0.002762458685903552, 'batch_size': 256}. Best is trial 24 with value: 7.56167952990336e-05.
Successful Trials:  47%|████▋     | 28/60 [1:22:30<1:54:09, 214.06s/trial][I 2024-06-04 16:12:24,372] Trial 28 finished with value: 9.40420586348898e-05 and parameters: {'hidden_size': 81, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.22471753415346546, 'optimizer': 'Adam', 'lr': 0.0036728295082663733, 'batch_size': 256}. Best is trial 24 with value: 7.56167952990336e-05.
Successful Trials:  48%|████▊     | 29/60 [1:26:08<1:51:12, 215.25s/trial][I 2024-06-04 16:16:04,715] Trial 29 finished with value: 6.779438958801621e-05 and parameters: {'hidden_size': 179, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1754360052435621, 'optimizer': 'Adam', 'lr': 0.006693895663929791, 'batch_size': 256}. Best is trial 29 with value: 6.779438958801621e-05.
Successful Trials:  50%|█████     | 30/60 [1:29:48<1:48:23, 216.78s/trial][I 2024-06-04 16:19:42,761] Trial 30 finished with value: 7.657334558899792e-05 and parameters: {'hidden_size': 170, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.17538237589159525, 'optimizer': 'Adam', 'lr': 0.006636065449024731, 'batch_size': 256}. Best is trial 29 with value: 6.779438958801621e-05.
Successful Trials:  52%|█████▏    | 31/60 [1:33:27<1:44:57, 217.16s/trial][I 2024-06-04 16:23:19,862] Trial 31 finished with value: 6.707696365213653e-05 and parameters: {'hidden_size': 179, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1813728557164148, 'optimizer': 'Adam', 'lr': 0.006604040491151783, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  53%|█████▎    | 32/60 [1:37:04<1:41:19, 217.14s/trial][I 2024-06-04 16:26:58,916] Trial 32 finished with value: 7.10984804821137e-05 and parameters: {'hidden_size': 189, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1827282972070025, 'optimizer': 'Adam', 'lr': 0.00651496409846058, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  55%|█████▌    | 33/60 [1:40:43<1:37:58, 217.71s/trial][I 2024-06-04 16:30:37,273] Trial 33 finished with value: 7.772099292582928e-05 and parameters: {'hidden_size': 193, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.251487550846889, 'optimizer': 'Adam', 'lr': 0.006665734832784255, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  57%|█████▋    | 34/60 [1:44:21<1:34:25, 217.91s/trial][I 2024-06-04 16:34:16,056] Trial 34 finished with value: 7.484783031417966e-05 and parameters: {'hidden_size': 190, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.2188643038033791, 'optimizer': 'Adam', 'lr': 0.008284231878447491, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  58%|█████▊    | 35/60 [1:48:00<1:30:54, 218.17s/trial][I 2024-06-04 16:37:54,656] Trial 35 finished with value: 7.44214359449305e-05 and parameters: {'hidden_size': 192, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.2241634867042525, 'optimizer': 'Adam', 'lr': 0.008212345512537998, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  60%|██████    | 36/60 [1:51:38<1:27:19, 218.30s/trial][I 2024-06-04 16:41:31,489] Trial 36 finished with value: 8.969912503062586e-05 and parameters: {'hidden_size': 213, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.23358941641020542, 'optimizer': 'Adam', 'lr': 0.007944921856937907, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  62%|██████▏   | 37/60 [1:55:15<1:23:30, 217.86s/trial][I 2024-06-04 16:45:08,479] Trial 37 finished with value: 7.862567753877752e-05 and parameters: {'hidden_size': 182, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.25764594968211374, 'optimizer': 'Adam', 'lr': 0.006425652059246607, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  63%|██████▎   | 38/60 [1:58:52<1:19:47, 217.60s/trial][I 2024-06-04 16:48:45,630] Trial 38 finished with value: 7.590980171745425e-05 and parameters: {'hidden_size': 147, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1868909843827583, 'optimizer': 'Adam', 'lr': 0.007336791720905157, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  65%|██████▌   | 39/60 [2:02:29<1:16:06, 217.46s/trial][I 2024-06-04 16:52:24,800] Trial 39 finished with value: 7.613752494067098e-05 and parameters: {'hidden_size': 290, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.20733842864880084, 'optimizer': 'Adam', 'lr': 0.005257093493381794, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  67%|██████▋   | 40/60 [2:06:09<1:12:39, 217.98s/trial][I 2024-06-04 16:56:03,890] Trial 40 finished with value: 8.272415280885361e-05 and parameters: {'hidden_size': 205, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.2725754552118155, 'optimizer': 'Adam', 'lr': 0.00871645422589195, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  68%|██████▊   | 41/60 [2:09:48<1:09:07, 218.31s/trial][I 2024-06-04 16:59:42,522] Trial 41 finished with value: 7.443297358246299e-05 and parameters: {'hidden_size': 183, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.2069171036467888, 'optimizer': 'Adam', 'lr': 0.008223485560012405, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  70%|███████   | 42/60 [2:13:26<1:05:31, 218.41s/trial][I 2024-06-04 17:03:20,767] Trial 42 finished with value: 8.160322358494537e-05 and parameters: {'hidden_size': 172, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.20605519224791494, 'optimizer': 'Adam', 'lr': 0.00621641477042384, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  72%|███████▏  | 43/60 [2:17:05<1:01:52, 218.36s/trial][I 2024-06-04 17:06:57,772] Trial 43 finished with value: 7.527235725622159e-05 and parameters: {'hidden_size': 239, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.18128535881647823, 'optimizer': 'Adam', 'lr': 0.007862012098926879, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  73%|███████▎  | 44/60 [2:20:42<58:07, 217.95s/trial]  [I 2024-06-04 17:09:19,773] Trial 44 finished with value: 0.00021347434312516192 and parameters: {'hidden_size': 151, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.16081112054062957, 'optimizer': 'Adam', 'lr': 0.0069822553853711895, 'batch_size': 2048}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  75%|███████▌  | 45/60 [2:23:04<48:47, 195.17s/trial][I 2024-06-04 17:12:49,430] Trial 45 finished with value: 0.0001301874821611909 and parameters: {'hidden_size': 221, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.23730819219940325, 'optimizer': 'RMSprop', 'lr': 0.009975690138587099, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  77%|███████▋  | 46/60 [2:26:33<46:33, 199.51s/trial][I 2024-06-04 17:16:37,161] Trial 46 finished with value: 7.152337583382578e-05 and parameters: {'hidden_size': 182, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1931750083490477, 'optimizer': 'Adam', 'lr': 0.008672915159005562, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  78%|███████▊  | 47/60 [2:30:21<45:03, 207.98s/trial][I 2024-06-04 17:19:22,125] Trial 47 finished with value: 0.00014189652154253247 and parameters: {'hidden_size': 209, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.19464320105744357, 'optimizer': 'Adam', 'lr': 0.009044596389586067, 'batch_size': 512}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  80%|████████  | 48/60 [2:33:06<39:00, 195.07s/trial][I 2024-06-04 17:21:46,560] Trial 48 finished with value: 0.00021840861160426914 and parameters: {'hidden_size': 134, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.17402502217010488, 'optimizer': 'RMSprop', 'lr': 0.006013135861840439, 'batch_size': 2048}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  82%|████████▏ | 49/60 [2:35:30<32:58, 179.88s/trial][I 2024-06-04 17:25:30,131] Trial 49 finished with value: 7.944172561857135e-05 and parameters: {'hidden_size': 165, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.2182433559453238, 'optimizer': 'Adam', 'lr': 0.007574398992556914, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  83%|████████▎ | 50/60 [2:39:14<32:09, 192.99s/trial][I 2024-06-04 17:27:55,036] Trial 50 finished with value: 0.003296951727594316 and parameters: {'hidden_size': 351, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.16389003851726167, 'optimizer': 'SGD', 'lr': 0.009373670642349483, 'batch_size': 1024}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  85%|████████▌ | 51/60 [2:41:39<26:47, 178.56s/trial][I 2024-06-04 17:31:33,400] Trial 51 finished with value: 7.53987225858123e-05 and parameters: {'hidden_size': 184, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.21244877964160366, 'optimizer': 'Adam', 'lr': 0.008331634393345867, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  87%|████████▋ | 52/60 [2:45:17<25:24, 190.50s/trial][I 2024-06-04 17:35:15,762] Trial 52 finished with value: 7.86476795899739e-05 and parameters: {'hidden_size': 181, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.19855354321294086, 'optimizer': 'Adam', 'lr': 0.008345373482438143, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  88%|████████▊ | 53/60 [2:49:00<23:20, 200.06s/trial][I 2024-06-04 17:38:58,106] Trial 53 finished with value: 8.278097437309175e-05 and parameters: {'hidden_size': 159, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.2265308208056409, 'optimizer': 'Adam', 'lr': 0.00700338647811201, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  90%|█████████ | 54/60 [2:52:42<20:40, 206.75s/trial][I 2024-06-04 17:42:41,009] Trial 54 finished with value: 7.434027419691128e-05 and parameters: {'hidden_size': 203, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.1884254478237803, 'optimizer': 'Adam', 'lr': 0.008769192452872058, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  92%|█████████▏| 55/60 [2:56:25<17:37, 211.59s/trial][I 2024-06-04 17:46:24,482] Trial 55 finished with value: 7.18377808936003e-05 and parameters: {'hidden_size': 247, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.18402275469716534, 'optimizer': 'Adam', 'lr': 0.008792185702610658, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  93%|█████████▎| 56/60 [3:00:08<14:20, 215.16s/trial][I 2024-06-04 17:49:11,984] Trial 56 finished with value: 8.648859743504739e-05 and parameters: {'hidden_size': 247, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.18336223886231695, 'optimizer': 'Adam', 'lr': 0.008772279449866499, 'batch_size': 512}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  95%|█████████▌| 57/60 [3:02:56<10:02, 200.86s/trial][I 2024-06-04 17:51:47,377] Trial 57 finished with value: 0.6824814750527849 and parameters: {'hidden_size': 268, 'activation': 'tanh', 'initialization': 'glorot', 'dropout': 0.15482726473922176, 'optimizer': 'RMSprop', 'lr': 0.009111350222089438, 'batch_size': 2048}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  97%|█████████▋| 58/60 [3:05:31<06:14, 187.22s/trial][I 2024-06-04 17:55:11,869] Trial 58 finished with value: 0.003897336005348453 and parameters: {'hidden_size': 229, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.17015670326017046, 'optimizer': 'SGD', 'lr': 0.0055919665861783666, 'batch_size': 256}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials:  98%|█████████▊| 59/60 [3:08:56<03:12, 192.40s/trial][I 2024-06-04 17:57:35,681] Trial 59 finished with value: 0.00012260021197163247 and parameters: {'hidden_size': 284, 'activation': 'sigmoid', 'initialization': 'glorot', 'dropout': 0.19075736553006872, 'optimizer': 'Adam', 'lr': 0.009436595208632433, 'batch_size': 1024}. Best is trial 31 with value: 6.707696365213653e-05.
Successful Trials: 100%|██████████| 60/60 [3:11:19<00:00, 191.33s/trial]
Best trial: 31
  Value: 6.707696365213653e-05
  Params: 
    hidden_size: 179
    activation: sigmoid
    initialization: glorot
    dropout: 0.1813728557164148
    optimizer: Adam
    lr: 0.006604040491151783
    batch_size: 256
Number of successful trials: 60
```
