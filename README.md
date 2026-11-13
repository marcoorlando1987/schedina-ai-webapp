# Schedina AI

Probabilità 1X2, gol attesi e Over 2.5 per Serie A, Premier League, La Liga, Bundesliga e Ligue 1.
Strumento a scopo informativo/statistico: il calcio è molto aleatorio e nessun modello garantisce risultati.

## Come funziona (v2)

```
football-data.co.uk (CSV storici + quote)      football-data.org (calendario)
            │                                           │
            └────────────►  job offline  ◄──────────────┘
                     python -m schedina_ai run
        feature (Elo, forma, gol) → walk-forward → modello → SQLite
                                                       │
                                        app.py (Streamlit, sola lettura)
```

* **Il report è istantaneo**: l'app non scarica né addestra nulla, legge solo il DB scritto dal job.
* **Feature reali** (prima c'erano solo ID numerici di squadra): Elo con regressione a inizio stagione,
  medie gol fatti/subiti pesate, forma ultime 5, giorni di riposo. Calcolate **senza leakage**
  (ogni partita usa solo lo stato precedente; c'è un test dedicato).
* **Quattro modelli in gara** (baseline, regressione logistica, XGBoost, Poisson). Ogni notte il job li
  valuta con **walk-forward** (si addestra sul passato, si testa sulla stagione successiva) e usa il
  migliore per log-loss.
* **Benchmark con le quote dei bookmaker**: i CSV le contengono. Se il modello non le batte in log-loss,
  la tab "Affidabilità" lo dice chiaramente.
* **Track record reale**: le predizioni vengono congelate al calcio d'inizio e confrontate col risultato.
* **Errori espliciti**: token errato, rate limit, fonte dati giù → il job fallisce con un messaggio
  chiaro (prima: "nessuna partita trovata").
* Nomi squadra mappati con alias espliciti + fuzzy match **dentro la stessa lega**; le squadre non
  riconosciute (neopromosse) sono marcate con ⚠️ e loggate.

## Uso locale

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

export FOOTBALL_DATA_API_TOKEN=...        # token gratuito di football-data.org
python -m schedina_ai evaluate            # solo validazione: guarda queste metriche per prime
python -m schedina_ai run                 # scarica, valuta, addestra, salva in data/schedina.db
streamlit run app.py
pytest                                    # test (dati sintetici, nessuna rete)
```

Opzioni di `run`: `--days-ahead 7`, `--days-back 7`, `--skip-eval`, `--model {baseline,logreg,xgb,poisson}`.
I dati vanno in `data/` (o `SCHEDINA_DATA_DIR`). I CSV delle stagioni chiuse non vengono riscaricati.

## Pubblicazione gratuita: Streamlit Community Cloud + GitHub Actions (consigliata per partire)

Costo zero. L'app è su Streamlit Community Cloud; un workflow di GitHub Actions aggiorna le previsioni
ogni notte e pubblica il DB sul branch `data`, che l'app legge via HTTPS.
Il repo deve essere **pubblico** (niente segreti nel codice: il token sta nei GitHub Secrets).

1. **Unisci il codice in `main`** (i workflow `schedule` girano solo dal branch di default).
2. **Aggiungi il secret**: repo → Settings → Secrets and variables → Actions → `FOOTBALL_DATA_API_TOKEN`.
3. **Primo popolamento**: tab Actions → "Aggiorna previsioni" → *Run workflow*. Crea il branch `data`
   con `schedina.db` (un solo commit, sovrascritto a ogni giro: il repo non cresce e `main` non cambia,
   quindi l'app non viene ridistribuita ogni giorno). Poi gira da solo alle 05:17 UTC.
4. **Deploy dell'app**: share.streamlit.io → *New app* → questo repo, branch `main`, file `app.py`,
   Python 3.12 (Advanced settings). Streamlit installa `requirements.txt`, che è volutamente leggero
   (niente scikit-learn/xgboost: l'app legge soltanto). Senza altra configurazione l'app scarica il DB da
   `https://raw.githubusercontent.com/marcoorlando1987/schedina-ai-webapp/data/schedina.db`
   (`DEFAULT_DB_URL` in `schedina_ai/config.py`; se fai un fork imposta `SCHEDINA_DB_URL` o modifica la costante).
5. Per i test c'è un secondo workflow (`tests.yml`) su push e pull request.

Da sapere:
* L'app ha una cache di 5 minuti sul DB, e raw.githubusercontent.com ne ha una sua di pochi minuti.
* Streamlit Community Cloud manda in sleep le app senza traffico (12 ore) e **non supporta domini
  personalizzati**: per quelli vedi la sezione Cloud Run.
* GitHub disattiva i workflow pianificati dopo 60 giorni senza attività nel repo; i push notturni sul
  branch `data` dovrebbero bastare, altrimenti li riattivi dalla tab Actions. Gli orari possono slittare
  di qualche decina di minuti.
* Se un giro fallisce (token, API, fonte dati giù) il DB precedente resta pubblicato e GitHub ti manda
  l'email di errore; l'app mostra sempre "Ultimo aggiornamento".

## Alternativa: Cloud Run (dominio proprio)

Stessa immagine per app e job; i dati stanno in un bucket GCS montato su `/data`.
Verifica i flag con `gcloud ... --help`: possono cambiare.

```bash
PROJECT=...; REGION=europe-west8; BUCKET=$PROJECT-schedina-data
IMAGE=$REGION-docker.pkg.dev/$PROJECT/schedina/app:latest

gcloud artifacts repositories create schedina --repository-format=docker --location=$REGION
gcloud builds submit --tag $IMAGE .
gcloud storage buckets create gs://$BUCKET --location=$REGION
printf '%s' "IL_TUO_TOKEN" | gcloud secrets create football-data-token --data-file=-

# 1) Job di aggiornamento (python -m schedina_ai run)
gcloud run jobs create schedina-job --image $IMAGE --region $REGION \
  --command python --args=-m,schedina_ai,run \
  --set-secrets FOOTBALL_DATA_API_TOKEN=football-data-token:latest \
  --add-volume name=data,type=cloud-storage,bucket=$BUCKET \
  --add-volume-mount volume=data,mount-path=/data \
  --memory 1Gi --task-timeout 15m

# 2) Schedulazione notturna (l'account di servizio deve poter eseguire il job)
gcloud scheduler jobs create http schedina-nightly --location $REGION \
  --schedule "0 6 * * *" --time-zone Europe/Rome --http-method POST \
  --uri "https://$REGION-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/schedina-job:run" \
  --oauth-service-account-email SA@$PROJECT.iam.gserviceaccount.com

# 3) App web (non ha bisogno del token)
gcloud run deploy schedina-app --image $IMAGE --region $REGION --allow-unauthenticated \
  --memory 512Mi --session-affinity \
  --add-volume name=data,type=cloud-storage,bucket=$BUCKET,readonly=true \
  --add-volume-mount volume=data,mount-path=/data

gcloud run jobs execute schedina-job --region $REGION --wait   # primo popolamento
```

Dominio personalizzato: mapping di dominio di Cloud Run (se la regione lo supporta) oppure
Load Balancer HTTPS con NEG serverless.

## Limiti da conoscere

* I CSV di football-data.co.uk si aggiornano con qualche ora/giorno di ritardo: l'Elo non vede
  le ultimissime partite.
* L'indipendenza dei gol di Poisson sottostima un po' i pareggi.
* La tabella `ALIASES` in `schedina_ai/teams.py` è scritta a mano: se una squadra compare con ⚠️ senza
  essere neopromossa, aggiungi l'alias (il log del job elenca i nomi non riconosciuti).
* Il gioco d'azzardo ha limiti di legge sulla pubblicità in Italia: tieni il sito informativo, con
  disclaimer, senza link/promo di bookmaker.
