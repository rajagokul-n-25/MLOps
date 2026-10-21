# 🦀 RustFS Object Storage (Local & Server S3)

High-performance, 100% open-source (Apache 2.0) S3-compatible object storage for DVC and MLflow.

---

## 🚀 Quick Start

### 1. Start RustFS
From this directory:
```bash
docker compose up -d
```
Or from the project root:
```bash
docker compose -f rustfs/docker-compose.yml up -d
```

### 2. Access Web Console
* **URL**: [http://localhost:9001](http://localhost:9001)
* **Access Key**: `admin`
* **Secret Key**: `admin@123`
* **Action**: Create a bucket named `house-prices`

### 3. S3 API Endpoint
* **S3 Endpoint**: `http://localhost:9000`

---

## 🔗 Connect DVC to RustFS

Run these commands in the project root:

```bash
# 1. Install S3 support
pip install dvc-s3

# 2. Add remote pointing to bucket
dvc remote add -d rustfs-s3 s3://house-prices/dvcstore

# 3. Configure endpoint and credentials
dvc remote modify rustfs-s3 endpointurl http://localhost:9000
dvc remote modify rustfs-s3 access_key_id admin
dvc remote modify rustfs-s3 secret_access_key admin@123

# 4. Push dataset to RustFS
dvc push
```

---

## 🛑 Stop RustFS
```bash
docker compose down
```
