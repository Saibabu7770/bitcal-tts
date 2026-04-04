# Security

## Supported versions

We aim to fix security issues in the **latest commit on `main`** for this research repository. There is no long-term LTS release line yet.

## Reporting a vulnerability

Please **do not** open a public issue for security-sensitive reports.

**Preferred:** use [GitHub Security Advisories](https://github.com/Saibabu7770/bitcal-tts/security/advisories/new) (Repository → **Security** → **Report a vulnerability**), if enabled for this repo.

**Alternative:** contact the repository owner via GitHub with a brief description and steps to reproduce.

We will acknowledge receipt when possible. This is a small research project; response times are best-effort.

## Scope

- **In scope:** dependency vulnerabilities, unsafe defaults in this codebase, credential handling in documented scripts.
- **Out of scope:** model safety / misuse of LLMs (follow your institution’s policies), third-party model weights, and issues in upstream PyTorch / Hugging Face beyond pinning versions in this repo.
