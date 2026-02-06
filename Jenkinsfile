pipeline {
    agent any

    environment {
        GCP_PROJECT = 'gen-lang-client-0068855436'
        REGION = 'us-central1'
        SERVICE_NAME = 'multi-doc-chat'
        REGISTRY = 'us-central1-docker.pkg.dev'
        REPOSITORY = 'cloud-run'
        IMAGE = "${REGISTRY}/${GCP_PROJECT}/${REPOSITORY}/${SERVICE_NAME}:latest"
        GCLOUD_PATH = '/usr/bin'   // gcloud is already on PATH in your Jenkins image
    }

    stages {

        stage("Clone from GitHub") {
            steps {
                checkout scmGit(
                    branches: [[name: '*/main']],
                    userRemoteConfigs: [[
                        credentialsId: 'github-token',
                        url: 'https://github.com/kaustubh-2187/RAG-end-to-end.git'
                    ]]
                )
            }
        }

        stage("Build Docker Image") {
            steps {
                sh '''
                docker build -t ${IMAGE} .
                '''
            }
        }

        stage("Push to Artifact Registry") {
            steps {
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]) {
                    sh '''
                    gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                    gcloud config set project ${GCP_PROJECT}

                    gcloud auth configure-docker ${REGISTRY} --quiet
                    docker push ${IMAGE}
                    '''
                }
            }
        }

        stage("Deploy to Cloud Run") {
            steps {
                withCredentials([
                    file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS'),
                    string(credentialsId: 'GROQ_API_KEY', variable: 'GROQ_KEY'),
                    string(credentialsId: 'GOOGLE_API_KEY', variable: 'GOOGLE_KEY'),
                    string(credentialsId: 'HF_TOKEN', variable: 'HF_TOKEN')
                ]) {
                    sh '''
                    gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}
                    gcloud config set project ${GCP_PROJECT}

                    gcloud run deploy ${SERVICE_NAME} \
                      --image=${IMAGE} \
                      --platform=managed \
                      --region=${REGION} \
                      --allow-unauthenticated \
                      --port=8000 \
                      --memory=2Gi \
                      --cpu=2 \
                      --timeout=300 \
                      --set-env-vars="GROQ_API_KEY=${GROQ_KEY},GOOGLE_API_KEY=${GOOGLE_KEY},HF_TOKEN=${HF_TOKEN},ENV=production"
                    '''
                }
            }
        }
    }

    post {
        success {
            echo '✅ Cloud Run deployment successful!'
        }
        failure {
            echo '❌ Cloud Run deployment failed!'
        }
    }
}
