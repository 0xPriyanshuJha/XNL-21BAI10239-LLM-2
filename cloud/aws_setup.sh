set -e


if [ -f .env ]; then
  echo "🔍 Loading environment variables from .env file..."
  export $(grep -v '^#' .env | xargs)
else
  echo "Error: .env file not found!"
  exit 1
fi

echo "Setting up AWS Environment for LLM Fine-Tuning..."


echo "🛠 Creating S3 bucket: $S3_BUCKET_NAME..."
aws s3 mb s3://$S3_BUCKET_NAME --region $AWS_REGION


echo "Creating IAM Role: $IAM_ROLE_NAME..."
aws iam create-role --role-name $IAM_ROLE_NAME --assume-role-policy-document file://iam_policy.json
aws iam attach-role-policy --role-name $IAM_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name $IAM_ROLE_NAME --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess


echo "Setting up Security Group: $SECURITY_GROUP..."
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name $SECURITY_GROUP --description "Security Group for LLM Training" --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 22 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP_ID --protocol tcp --port 8888 --cidr 0.0.0.0/0

echo "Launching EC2 instance ($INSTANCE_TYPE)..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-08e5424edfe926b43 \
  --instance-type $INSTANCE_TYPE \
  --key-name $KEY_NAME \
  --security-group-ids $SECURITY_GROUP_ID \
  --iam-instance-profile Name=$IAM_ROLE_NAME \
  --query 'Instances[0].InstanceId' --output text)

echo "Waiting for instance to start..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

echo "EC2 Instance Launched! ID: $INSTANCE_ID"
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "🔗 Access your instance: ssh -i $KEY_NAME.pem ubuntu@$PUBLIC_IP"


echo "🛠 Configuring MLflow remote tracking..."
echo "export MLFLOW_TRACKING_URI=$MLFLOW_BUCKET" >> ~/.bashrc
source ~/.bashrc


echo "Installing required libraries on EC2..."
ssh -i "$KEY_NAME.pem" ubuntu@$PUBLIC_IP <<EOF
  sudo apt update && sudo apt upgrade -y
  sudo apt install -y python3-pip awscli unzip
  pip install mlflow boto3 transformers torch deepspeed
EOF

echo "AWS Setup Complete!"
