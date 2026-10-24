# Grounding Engine

Grounding Engine is the backend service for **Grounded**, a Retrieval-Augmented Generation platform for answering questions over user-owned documents with citations, evidence traces, and tiered verification.

This repository contains the API, ingestion pipeline, retrieval system, verification logic, background workers, database migrations, benchmark tooling, and backend tests. The frontend/client application is maintained separately.

## What This Backend Does

Grounding Engine turns uploaded documents into queryable, traceable knowledge bases.

- Manages tenants, workspaces, datasets, agents, conversations, API keys, and team members
- Uploads, extracts, chunks, stores, and indexes documents
- Runs sparse + dense hybrid retrieval over indexed content
- Packages evidence for grounded answer generation
- Produces cited answers with response shaping and degraded behavior when evidence is weak
- Records query traces, run history, feedback, audit logs, and dashboard summaries
- Supports async processing for longer-running high-assurance workflows
- Provides benchmark fixtures, reports, raw outputs, and evaluation tooling

## Execution Model

The backend supports three execution tiers:

| Tier | Backend Behavior |
| --- | --- |
| Standard | Fast baseline RAG with hybrid retrieval, citations, evidence packaging, and traces |
| Enterprise | Stronger retrieval precision with reranking, freshness scoring, and richer evidence selection |
| Critical | High-assurance verification with contradiction handling, corrective retry, and async runs |

## Architecture At A Glance

```text
Client / API consumer
        |
        v
FastAPI routes
        |
        v
Services: auth, datasets, documents, agents, conversations, query, runs
        |
        v
Pipeline: extraction -> chunking -> sparse/dense indexing -> retrieval -> generation -> verification
        |
        v
PostgreSQL + Qdrant + Redis + S3-compatible storage
```

## Tech Stack

| Area | Tools |
| --- | --- |
| API | FastAPI, Pydantic |
| Database | PostgreSQL, SQLAlchemy, Alembic |
| Retrieval | Qdrant, sparse indexing, dense embeddings, reranking hooks |
| Workers | Redis, ARQ |
| Storage | MinIO / S3-compatible object storage |
| Testing | pytest |
| Evaluation | Custom benchmark runner, benchmark fixtures, BEIR/NFCorpus support |

## Repository Layout

```text
backend/
  app/
    api/                FastAPI routes and dependencies
    core/               Shared infrastructure, security, retrieval clients
    models/             SQLAlchemy models
    pipeline/           Ingestion and query pipeline orchestration
    schemas/            Request and response schemas
    services/           Business logic
    workers/            Background worker entrypoints
  alembic/              Database migrations
  reports/              Benchmark reports, summaries, and raw outputs
  scripts/              Utility scripts
  tests/                Unit, integration, contract, load, and evaluation tests

docs/                   System design, API flow, product flow, benchmarks
.env.example            Public environment template
docker-compose.yml      Local infrastructure
docker-compose.dev.yml  Backend development stack
Makefile                Common backend commands
requirements.txt        Python dependencies
```

## Prerequisites

- Python 3.11+
- Docker Desktop
- Git

## Quick Start

Clone the repository:

```bash
git clone https://github.com/Grounded-RAG/grounding-engine.git
cd grounding-engine
```

Create a local environment file:

```bash
cp .env.example .env
```

Start local infrastructure:

```bash
docker compose up -d postgres redis qdrant minio
```

Create and activate a virtual environment:

```bash
python -m venv backend/.venv
source backend/.venv/bin/activate
```

On Windows PowerShell:

```powershell
backend\.venv\Scripts\Activate.ps1
```

Install dependencies and apply migrations:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd backend
python -m alembic upgrade head
```

Run the API:

```bash
python -m uvicorn app.main:app --reload
```

Local endpoints:

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health/ready`
- MinIO console: `http://localhost:9001`

## Common Commands

```bash
make infra-up        # Start PostgreSQL, Redis, Qdrant, and MinIO
make migrate         # Apply database migrations
make test            # Run backend tests
make ci              # Run compile, migration, and foundation test checks
make infra-down      # Stop local infrastructure
```

Run tests directly:

```bash
cd backend
pytest
```

Run a compile check:

```bash
python -m compileall backend/app backend/tests backend/alembic
```

## Benchmarks

Benchmark assets are intentionally included for evaluation transparency.

Key locations:

- `backend/tests/evaluation/`: benchmark runner, schema, variants, fixtures, and tests
- `backend/reports/benchmark-current/`: current benchmark report, summary, and raw results
- `backend/reports/benchmark-beir-nfcorpus/`: BEIR/NFCorpus report, summary, queries, and raw results
- `docs/BENCHMARKING_README.md`: benchmark usage notes
- `docs/BENCHMARK_AND_PERFORMANCE_OPTIMIZATION_PLAN.md`: evaluation and optimization plan

Run the BEIR/NFCorpus benchmark when the dataset is available locally:

```bash
make benchmark-beir
```

Run a smaller smoke benchmark:

```bash
make benchmark-beir-smoke
```

## Documentation

- `docs/SYSTEM_DESIGN.md`: backend architecture, runtime flow, and tier behavior
- `docs/PRODUCT_FLOW.md`: product objects and backend-supported workflows
- `docs/SWAGGER_TEST_FLOW.md`: manual API testing flow
- `docs/BENCHMARKING_README.md`: benchmark setup and interpretation
- `docs/BENCHMARK_ISSUE_BACKLOG.md`: known benchmark gaps and follow-up work

## Environment And Secrets

Use `.env.example` as the public template. Do not commit local `.env` files, real API keys, provider secrets, database credentials, or private tokens.

`FRONTEND_URL` is still a backend setting because OAuth, email, and invitation flows need the URL of the separately hosted client application.

## AWS Bedrock

Every provider slot (generation, dense embeddings, the Enterprise reranker) has a `bedrock_v1` backend that
runs on AWS Bedrock through boto3, which the backend already depends on for object storage.

- Credentials come from the AWS default chain: `AWS_PROFILE`, or `AWS_ACCESS_KEY_ID` and
  `AWS_SECRET_ACCESS_KEY` in the shell, or an instance role. The S3 settings are not used.
- `BEDROCK_REGION` picks the region; `BEDROCK_MODEL` is the generation and reranking model, given as an
  inference profile id (for example `us.anthropic.<model>`).
- `BEDROCK_EMBEDDING_MODEL` defaults to Titan Text Embeddings V2 (`amazon.titan-embed-text-v2:0`), whose
  request body the embedding backend speaks. Its vector size is 256, 512 or 1024, so set
  `DENSE_EMBEDDING_DIMENSIONS` to one of those and re-index after switching embedding backends.
- Generation uses the Converse API with the same grounded JSON contract, prompt and evidence validation as the
  Gemini backend, so answers from the two providers are held to the same rules.

```bash
GENERATOR_BACKEND=bedrock_v1
EMBEDDING_BACKEND=bedrock_v1
ENTERPRISE_RERANKER_BACKEND=bedrock_v1
BEDROCK_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.<model>
DENSE_EMBEDDING_DIMENSIONS=1024
```

The unit tests in `backend/tests/unit/test_bedrock_providers.py` cover the three backends against a fake
client and need no AWS account.
