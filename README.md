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

## Pubblicazione su Cloud Run (consigliata)

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
Alternativa gratuita e immediata: **Streamlit Community Cloud** (indirizzo `*.streamlit.app`); in quel caso
committa a mano (`git add -f data/schedina.db`) un DB generato in locale oppure aggiorna il DB con un workflow GitHub Actions
pianificato.

## Limiti da conoscere

* I CSV di football-data.co.uk si aggiornano con qualche ora/giorno di ritardo: l'Elo non vede
  le ultimissime partite.
* L'indipendenza dei gol di Poisson sottostima un po' i pareggi.
* La tabella `ALIASES` in `schedina_ai/teams.py` è scritta a mano: se una squadra compare con ⚠️ senza
  essere neopromossa, aggiungi l'alias (il log del job elenca i nomi non riconosciuti).
* Il gioco d'azzardo ha limiti di legge sulla pubblicità in Italia: tieni il sito informativo, con
  disclaimer, senza link/promo di bookmaker.
