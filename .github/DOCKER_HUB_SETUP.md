# GitHub Actions Docker Hub Setup

This repository includes a GitHub Actions workflow that automatically builds and pushes Docker images to Docker Hub.

## Setup Instructions

### 1. Create Docker Hub Access Token

1. Log in to [Docker Hub](https://hub.docker.com/)
2. Go to **Account Settings** → **Security** → **Access Tokens**
3. Click **New Access Token**
4. Give it a description (e.g., "GitHub Actions")
5. Set permissions to **Read, Write, Delete**
6. Click **Generate**
7. **Copy the token** (you won't be able to see it again)

### 2. Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add the following secrets:

   - **Name:** `DOCKERHUB_USERNAME`
     - **Value:** Your Docker Hub username
   
   - **Name:** `DOCKERHUB_TOKEN`
     - **Value:** The access token you generated in step 1

### 3. Workflow Triggers

The workflow automatically runs on:

- **Push to main/master branch:** Builds and pushes with `latest` tag and branch name
- **Push tags (v*):** Builds and pushes with semantic version tags (e.g., `v1.0.0`, `v1.0`, `v1`)
- **Pull requests:** Builds only (does not push)
- **Manual trigger:** Can be triggered manually from the Actions tab

### 4. Docker Image Tags

The workflow creates the following tags:

- `latest` - Latest build from the default branch
- `main` or `master` - Latest build from that branch
- `v1.0.0`, `v1.0`, `v1` - Semantic version tags (when pushing version tags)
- `main-<sha>` - Branch name with commit SHA
- `pr-<number>` - Pull request builds (not pushed)

### 5. Using the Built Images

After the workflow runs successfully, your image will be available at:

```
docker pull <your-dockerhub-username>/vllm-inference-api:latest
```

Or with specific tags:

```
docker pull <your-dockerhub-username>/vllm-inference-api:v1.0.0
docker pull <your-dockerhub-username>/vllm-inference-api:main
```

### 6. Manual Workflow Trigger

To manually trigger the workflow:

1. Go to **Actions** tab in your repository
2. Select **Build and Push Docker Image** workflow
3. Click **Run workflow**
4. Select the branch and click **Run workflow**

## Troubleshooting

### Authentication Failed

- Verify `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set correctly
- Ensure the Docker Hub access token has not expired
- Check that the token has **Read, Write, Delete** permissions

### Build Failed

- Check the workflow logs in the Actions tab
- Verify the Dockerfile builds successfully locally
- Ensure all required files are committed to the repository

### Image Not Appearing on Docker Hub

- Verify the workflow completed successfully (green checkmark)
- Check that the event was not a pull request (PRs don't push)
- Confirm you're logged in to the correct Docker Hub account
