# =============================================================
#  BioIntelligence Platform — Makefile
# =============================================================
#  Usage:
#    make setup   — first-time initialisation
#    make up      — start all services
#    make down    — stop all services
#    make help    — list all targets
# =============================================================

SHELL := /bin/bash

# ANSI colour helpers
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
BOLD   := \033[1m
RESET  := \033[0m

COMPOSE         := docker compose
PROJECT         := biointel
BACKUP_DIR      := ./backups
TIMESTAMP       := $(shell date +%Y%m%d_%H%M%S)

.PHONY: help setup up down logs ps \
        db-shell airflow-shell \
        trigger-dag1 trigger-dag2 trigger-dag3 trigger-all \
        reset backup \
        deploy-railway deploy-gcp deploy-aws \
        health

# Default target
.DEFAULT_GOAL := help

# ------------------------------------------------------------------
#  help — list all documented targets
# ------------------------------------------------------------------
help:
	@echo -e "$(BOLD)$(CYAN)BioIntelligence Platform — Available Targets$(RESET)"
	@echo -e "$(CYAN)---------------------------------------------$(RESET)"
	@echo -e "  $(GREEN)setup$(RESET)          Copy .env.example → .env; create data/ and models/ dirs"
	@echo -e "  $(GREEN)up$(RESET)             Build images and start all services in the background"
	@echo -e "  $(GREEN)down$(RESET)           Stop and remove containers (volumes preserved)"
	@echo -e "  $(GREEN)logs$(RESET)           Tail logs from all services"
	@echo -e "  $(GREEN)ps$(RESET)             Show container status"
	@echo -e ""
	@echo -e "  $(YELLOW)db-shell$(RESET)       Open psql shell inside the Postgres container"
	@echo -e "  $(YELLOW)airflow-shell$(RESET)  Open bash inside the Airflow webserver container"
	@echo -e ""
	@echo -e "  $(YELLOW)trigger-dag1$(RESET)   Trigger genetics_qc_pca_pipeline"
	@echo -e "  $(YELLOW)trigger-dag2$(RESET)   Trigger mri_segmentation_pipeline"
	@echo -e "  $(YELLOW)trigger-dag3$(RESET)   Trigger dna_model_inference_pipeline"
	@echo -e "  $(YELLOW)trigger-all$(RESET)    Trigger all three pipelines in sequence"
	@echo -e ""
	@echo -e "  $(RED)reset$(RESET)          DANGER: stop containers AND delete all volumes / data"
	@echo -e "  $(GREEN)backup$(RESET)         Dump Postgres to ./backups/backup_TIMESTAMP.sql"
	@echo -e ""
	@echo -e "  $(CYAN)deploy-railway$(RESET) Print Railway deployment instructions"
	@echo -e "  $(CYAN)deploy-gcp$(RESET)     Run deploy/gcp_cloudrun.sh"
	@echo -e "  $(CYAN)deploy-aws$(RESET)     Run deploy/aws_ecs.sh"
	@echo -e ""
	@echo -e "  $(GREEN)health$(RESET)         Curl health endpoints for all services"

# ------------------------------------------------------------------
#  setup — first-time initialisation
# ------------------------------------------------------------------
setup:
	@echo -e "$(BOLD)$(CYAN)[setup]$(RESET) Initialising BioIntelligence Platform..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo -e "$(GREEN)  ✔ Copied .env.example → .env$(RESET)"; \
		echo -e "$(YELLOW)  ⚠  Edit .env and set all passwords / keys before running 'make up'$(RESET)"; \
	else \
		echo -e "$(YELLOW)  ℹ  .env already exists — skipping copy$(RESET)"; \
	fi
	@mkdir -p data models backups superset db dags scripts
	@echo -e "$(GREEN)  ✔ Created required directories: data/ models/ backups/ superset/ db/ dags/ scripts/$(RESET)"
	@if [ ! -f db/init.sql ]; then \
		echo -e "-- BioIntelligence Platform — initial schema placeholder\n-- Replace with your actual DDL statements." > db/init.sql; \
		echo -e "$(GREEN)  ✔ Created placeholder db/init.sql$(RESET)"; \
	fi
	@if [ ! -f superset/superset_config.py ]; then \
		printf 'import os\n\nSECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "changeme")\nSQLALCHEMY_DATABASE_URI = (\n    "postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}".format(\n        user=os.environ.get("DATABASE_USER", "biointel"),\n        pw=os.environ.get("DATABASE_PASSWORD", ""),\n        host=os.environ.get("DATABASE_HOST", "postgres"),\n        port=os.environ.get("DATABASE_PORT", "5432"),\n        db=os.environ.get("DATABASE_DB", "biointel"),\n    )\n)\nCACHE_CONFIG = {\n    "CACHE_TYPE": "RedisCache",\n    "CACHE_DEFAULT_TIMEOUT": 300,\n    "CACHE_KEY_PREFIX": "superset_",\n    "CACHE_REDIS_HOST": os.environ.get("REDIS_HOST", "redis"),\n    "CACHE_REDIS_PORT": int(os.environ.get("REDIS_PORT", "6379")),\n    "CACHE_REDIS_DB": 1,\n}\nDATA_CACHE_CONFIG = CACHE_CONFIG\nFEATURE_FLAGS = {"ALERT_REPORTS": True}\nROWS_PER_PAGE = 50\n' > superset/superset_config.py; \
		echo -e "$(GREEN)  ✔ Created superset/superset_config.py$(RESET)"; \
	fi
	@echo -e "$(BOLD)$(GREEN)[setup] Done. Run 'make up' to start the platform.$(RESET)"

# ------------------------------------------------------------------
#  up — build and start all services
# ------------------------------------------------------------------
up:
	@echo -e "$(BOLD)$(CYAN)[up]$(RESET) Building and starting all services..."
	@$(COMPOSE) up -d --build
	@echo -e "$(BOLD)$(GREEN)[up] Services started. Run 'make ps' to check status.$(RESET)"
	@echo -e "$(CYAN)  Airflow UI  → http://localhost:8080$(RESET)"
	@echo -e "$(CYAN)  Superset UI → http://localhost:8088$(RESET)"

# ------------------------------------------------------------------
#  down — stop all services
# ------------------------------------------------------------------
down:
	@echo -e "$(BOLD)$(YELLOW)[down]$(RESET) Stopping services..."
	@$(COMPOSE) down
	@echo -e "$(GREEN)[down] Done.$(RESET)"

# ------------------------------------------------------------------
#  logs — tail all service logs
# ------------------------------------------------------------------
logs:
	@$(COMPOSE) logs -f

# ------------------------------------------------------------------
#  ps — show container status
# ------------------------------------------------------------------
ps:
	@$(COMPOSE) ps

# ------------------------------------------------------------------
#  db-shell — interactive psql inside the Postgres container
# ------------------------------------------------------------------
db-shell:
	@echo -e "$(CYAN)[db-shell]$(RESET) Opening psql shell (Ctrl+D to exit)..."
	@docker exec -it biointel-postgres psql -U biointel -d biointel

# ------------------------------------------------------------------
#  airflow-shell — bash inside airflow-webserver
# ------------------------------------------------------------------
airflow-shell:
	@echo -e "$(CYAN)[airflow-shell]$(RESET) Opening bash (Ctrl+D to exit)..."
	@docker exec -it biointel-airflow-webserver bash

# ------------------------------------------------------------------
#  DAG triggers
# ------------------------------------------------------------------
trigger-dag1:
	@echo -e "$(CYAN)[trigger-dag1]$(RESET) Triggering genetics_qc_pca_pipeline..."
	@docker exec biointel-airflow-webserver airflow dags trigger genetics_qc_pca_pipeline
	@echo -e "$(GREEN)  ✔ genetics_qc_pca_pipeline triggered$(RESET)"

trigger-dag2:
	@echo -e "$(CYAN)[trigger-dag2]$(RESET) Triggering mri_segmentation_pipeline..."
	@docker exec biointel-airflow-webserver airflow dags trigger mri_segmentation_pipeline
	@echo -e "$(GREEN)  ✔ mri_segmentation_pipeline triggered$(RESET)"

trigger-dag3:
	@echo -e "$(CYAN)[trigger-dag3]$(RESET) Triggering dna_model_inference_pipeline..."
	@docker exec biointel-airflow-webserver airflow dags trigger dna_model_inference_pipeline
	@echo -e "$(GREEN)  ✔ dna_model_inference_pipeline triggered$(RESET)"

trigger-all: trigger-dag1 trigger-dag2 trigger-dag3
	@echo -e "$(BOLD)$(GREEN)[trigger-all] All three pipelines triggered.$(RESET)"

# ------------------------------------------------------------------
#  reset — DANGER: remove all containers AND volumes
# ------------------------------------------------------------------
reset:
	@echo -e "$(RED)$(BOLD)[reset] WARNING: This will delete ALL containers, volumes, and data!$(RESET)"
	@echo -e "$(RED)Press Ctrl+C within 5 seconds to abort...$(RESET)"
	@sleep 5
	@$(COMPOSE) down -v
	@echo -e "$(BOLD)$(YELLOW)[reset] All containers and volumes removed.$(RESET)"

# ------------------------------------------------------------------
#  backup — dump Postgres to ./backups/backup_TIMESTAMP.sql
# ------------------------------------------------------------------
backup:
	@mkdir -p $(BACKUP_DIR)
	@echo -e "$(CYAN)[backup]$(RESET) Dumping database to $(BACKUP_DIR)/backup_$(TIMESTAMP).sql ..."
	@docker exec biointel-postgres pg_dump \
		-U biointel \
		-d biointel \
		--no-password \
		> $(BACKUP_DIR)/backup_$(TIMESTAMP).sql
	@echo -e "$(GREEN)  ✔ Backup saved: $(BACKUP_DIR)/backup_$(TIMESTAMP).sql$(RESET)"

# ------------------------------------------------------------------
#  deploy-railway — print Railway deployment instructions
# ------------------------------------------------------------------
deploy-railway:
	@echo -e "$(BOLD)$(CYAN)Railway Deployment Instructions$(RESET)"
	@echo -e "$(CYAN)================================$(RESET)"
	@echo -e "1. Install the Railway CLI:  npm install -g @railway/cli"
	@echo -e "2. Login:                    railway login"
	@echo -e "3. Link this project:        railway link"
	@echo -e "4. Set secrets via Railway dashboard or CLI:"
	@echo -e "     railway variables set POSTGRES_PASSWORD=<value>"
	@echo -e "     railway variables set AIRFLOW_ADMIN_PASSWORD=<value>"
	@echo -e "     railway variables set AIRFLOW__CORE__FERNET_KEY=<value>"
	@echo -e "     railway variables set SUPERSET_SECRET_KEY=<value>"
	@echo -e "     railway variables set SUPERSET_ADMIN_PASSWORD=<value>"
	@echo -e "5. Deploy:                   railway up"
	@echo -e "$(YELLOW)See deploy/railway.toml for service configuration.$(RESET)"

# ------------------------------------------------------------------
#  deploy-gcp — run the GCP Cloud Run deployment script
# ------------------------------------------------------------------
deploy-gcp:
	@echo -e "$(CYAN)[deploy-gcp]$(RESET) Running GCP Cloud Run deployment..."
	@bash deploy/gcp_cloudrun.sh

# ------------------------------------------------------------------
#  deploy-aws — run the AWS ECS deployment script
# ------------------------------------------------------------------
deploy-aws:
	@echo -e "$(CYAN)[deploy-aws]$(RESET) Running AWS ECS deployment..."
	@bash deploy/aws_ecs.sh

# ------------------------------------------------------------------
#  health — curl all service health endpoints
# ------------------------------------------------------------------
health:
	@echo -e "$(BOLD)$(CYAN)[health]$(RESET) Checking service health endpoints..."
	@echo -en "  Airflow webserver (http://localhost:8080/health)  → "
	@curl -sf http://localhost:8080/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('\033[0;32m' + d.get('status','unknown') + '\033[0m')" 2>/dev/null || echo -e "$(RED)unreachable$(RESET)"
	@echo -en "  Superset        (http://localhost:8088/health)    → "
	@curl -sf http://localhost:8088/health | python3 -c "import sys,json; d=json.load(sys.stdin); print('\033[0;32m' + d.get('status','unknown') + '\033[0m')" 2>/dev/null || echo -e "$(RED)unreachable$(RESET)"
	@echo -en "  PostgreSQL      (pg_isready)                      → "
	@docker exec biointel-postgres pg_isready -U biointel -d biointel > /dev/null 2>&1 && echo -e "$(GREEN)accepting connections$(RESET)" || echo -e "$(RED)not ready$(RESET)"
	@echo -en "  Redis           (redis-cli ping)                   → "
	@docker exec biointel-redis redis-cli ping 2>/dev/null | grep -q PONG && echo -e "$(GREEN)PONG$(RESET)" || echo -e "$(RED)unreachable$(RESET)"
