# ML-Powered Classification App with DevOps & MLOps Integration
# 基于机器学习的分类应用 - DevOps与MLops集成

## Project Overview / 项目概述

This project demonstrates a complete DevOps and MLOps workflow by building a machine learning-powered classification application. The app uses Support Vector Machine (SVM) to perform binary classification tasks, with comprehensive data preprocessing, model training, and evaluation capabilities.

本项目通过构建一个基于机器学习的分类应用，展示了完整的DevOps和MLops工作流程。该应用使用支持向量机(SVM)执行二分类任务，具备完整的数据预处理、模型训练和评估功能。

## Application Functionality / 应用功能

### Core Features / 核心功能

The `app.py` implements a complete machine learning pipeline:

`app.py`实现了完整的机器学习流水线：

1. **Data Preprocessing / 数据预处理**:
   - Removes unnecessary zero columns (删除不必要的零列)
   - Handles missing values with forward filling (使用前向填充处理缺失值)
   - Separates features and target variables (分离特征和目标变量)

2. **Model Training / 模型训练**:
   - Uses SVM with RBF kernel (使用RBF核的SVM)
   - Implements data standardization (实现数据标准化)
   - Configurable hyperparameters (可配置的超参数)

3. **Model Evaluation / 模型评估**:
   - Calculates accuracy, precision, recall, and F1-score (计算准确率、精确率、召回率和F1分数)
   - Generates confusion matrices (生成混淆矩阵)
   - Provides detailed classification reports (提供详细的分类报告)

4. **MLflow Integration / MLflow集成**:
   - Tracks experiments automatically (自动跟踪实验)
   - Logs hyperparameters and metrics (记录超参数和指标)
   - Saves trained models as artifacts (将训练好的模型保存为工件)

## DevOps Implementation / DevOps实现

### 1. Version Control / 版本控制

**Git Branching Strategy / Git分支策略**:
- `main`: Production-ready code (生产就绪代码)
- `staging`: Pre-production testing (预生产测试)
- `dev`: Development branch (开发分支)
- `feature/*`: Feature development branches (功能开发分支)

**Merging Path / 合并路径**:feature → dev → staging → main


### 2. Docker Containerization / Docker容器化

**Dockerfile Features / Dockerfile特性**:
- Uses Python 3.9 slim base image (使用Python 3.9精简基础镜像)
- Multi-stage build optimization (多阶段构建优化)
- Proper dependency management (适当的依赖管理)
- Environment variable configuration (环境变量配置)

**Key Docker Commands / 关键Docker命令**:
```bash
# Build the image / 构建镜像
docker build -t mlops-app:latest .

# Run the container / 运行容器
docker run --rm mlops-app:latest
```

### 3. CI/CD Pipeline / CI/CD流水线

**GitHub Actions Workflow / GitHub Actions工作流**:

The `.github/workflows/main.yml` implements a comprehensive CI/CD pipeline:

`.github/workflows/main.yml`实现了全面的CI/CD流水线：

#### CI Requirements (on every PR) / CI要求（每次PR时）:
1. **Code Quality Checks / 代码质量检查**:
   - Flake8 linting (Flake8代码检查)
   - Black formatting verification (Black格式验证)
   - Static analysis (静态分析)

2. **Testing / 测试**:
   - Full test suite execution (完整测试套件执行)
   - Edge case coverage (边界情况覆盖)
   - Integration testing (集成测试)

3. **Docker Build Verification / Docker构建验证**:
   - Image build success (镜像构建成功)
   - Runtime verification (运行时验证)
   - Container functionality test (容器功能测试)

#### CD Requirements / CD要求:

**Staging Deployment / 预发布部署**:
- Triggers on merges to `staging` branch (在合并到`staging`分支时触发)
- Uses staging-specific environment variables (使用预发布特定环境变量)
- Builds staging Docker image (构建预发布Docker镜像)

**Production Deployment / 生产部署**:
- Triggers on merges to `main` branch (在合并到`main`分支时触发)
- Uses production-specific environment variables (使用生产特定环境变量)
- Builds production Docker image (构建生产Docker镜像)

## MLOps Implementation / MLOps实现

### 1. Data Versioning with DVC / 使用DVC进行数据版本控制

**DVC Configuration / DVC配置**:
- Initialized DVC repository (初始化DVC仓库)
- Remote storage configured (配置远程存储)
- Data tracking with `.dvc` files (使用`.dvc`文件跟踪数据)

**Dataset Versions / 数据集版本**:
- **v1**: Initial usable dataset (初始可用数据集)
- **v2**: Improved/cleaned/augmented version (改进/清理/增强版本)

**Data Documentation / 数据文档**:
Located in `ml/README_data.md`:
- Data source information (数据源信息)
- Changes from v1 → v2 (从v1到v2的变化)
- Commit hash to dataset version mapping (提交哈希到数据集版本的映射)

### 2. Experiment Tracking with MLflow / 使用MLflow进行实验跟踪

**Logged Information / 记录信息**:
- **Code Version**: Git commit SHA (代码版本：Git提交SHA)
- **Dataset Version**: DVC dataset version ID (数据集版本：DVC数据集版本ID)
- **Hyperparameters**: C, gamma, kernel, test_size, random_state (超参数)
- **Metrics**: accuracy, recall, precision, f1_score (指标)
- **Artifacts**: Trained model weights (工件：训练好的模型权重)

**Experiment Workflow / 实验工作流**:
1. Training script automatically calls MLflow APIs (训练脚本自动调用MLflow API)
2. Logs parameters, metrics, and artifacts (记录参数、指标和工件)
3. Produces run history in local file backend (在本地文件后端生成运行历史)

**Tracked Experiments / 跟踪的实验**:
- **Baseline Model**: Initial SVM implementation (基线模型：初始SVM实现)
- **Improved Model**: Tuned hyperparameters (改进模型：调优的超参数)

## Testing Strategy / 测试策略

The project includes comprehensive testing with at least 3 meaningful test suites:

项目包含全面的测试，至少有3个有意义的测试套件：

### Test Coverage / 测试覆盖

1. **Data Loading Tests / 数据加载测试**:
   - File existence validation (文件存在性验证)
   - Data integrity checks (数据完整性检查)
   - Edge cases for missing files (缺失文件的边界情况)

2. **Data Preprocessing Tests / 数据预处理测试**:
   - Column removal validation (列删除验证)
   - Missing value handling (缺失值处理)
   - Feature-target separation (特征-目标分离)

3. **Model Training Tests / 模型训练测试**:
   - Model instantiation (模型实例化)
   - Training process validation (训练过程验证)
   - Prediction capability (预测能力)

4. **Model Evaluation Tests / 模型评估测试**:
   - Metric calculation accuracy (指标计算准确性)
   - Prediction output validation (预测输出验证)
   - Performance benchmarking (性能基准测试)

## Project Structure / 项目结构

```
FinalProject/
├── src/
│   └── app.py                 # Main application / 主应用
├── tests/
│   └── test_app.py           # Test suite / 测试套件
├── data/
│   ├── train_and_test2.csv   # Dataset / 数据集
│   └── train_and_test2.csv.dvc # DVC pointer / DVC指针
├── .github/
│   └── workflows/
│       └── main.yml          # CI/CD pipeline / CI/CD流水线
├── Dockerfile                # Container definition / 容器定义
├── requirements.txt          # Python dependencies / Python依赖
├── .gitignore               # Git ignore rules / Git忽略规则
├── .dockerignore            # Docker ignore rules / Docker忽略规则
└── .dvcignore              # DVC ignore rules / DVC忽略规则
```

## Usage Instructions / 使用说明

### Local Development / 本地开发

1. **Clone the repository / 克隆仓库**:
```bash
git clone <repository-url>
cd FinalProject
```

2. **Install dependencies / 安装依赖**:
```bash
pip install -r requirements.txt
```

3. **Run the application / 运行应用**:
```bash
python src/app.py
```

4. **Run tests / 运行测试**:
```bash
python -m pytest tests/ -v
```

### Docker Usage / Docker使用

1. **Build the image / 构建镜像**:
```bash
docker build -t mlops-app:latest .
```

2. **Run the container / 运行容器**:
```bash
docker run --rm mlops-app:latest
```

### DVC Data Management / DVC数据管理

1. **Check data status / 检查数据状态**:
```bash
dvc status
```

2. **Pull data from remote / 从远程拉取数据**:
```bash
dvc pull
```

### MLflow Experiment Tracking / MLflow实验跟踪

1. **View experiments / 查看实验**:
```bash
mlflow ui
```

2. **Access MLflow UI / 访问MLflow UI**:
Open `http://localhost:5000` in your browser (在浏览器中打开)

## Environment Configuration / 环境配置

The application uses environment variables for configuration:

应用使用环境变量进行配置：

- `MY_SECRET_KEY`: Secret key for application (应用密钥)
- `DATA_PATH`: Path to the dataset file (数据集文件路径)

Create a `.env` file for local development:

为本地开发创建`.env`文件：

```env
MY_SECRET_KEY=your_secret_key_here
DATA_PATH=data/train_and_test2.csv
```

## Performance Metrics / 性能指标

The application tracks the following ML metrics:

应用跟踪以下ML指标：

- **Accuracy**: Overall classification accuracy (整体分类准确率)
- **Precision**: True positive rate (精确率)
- **Recall**: Sensitivity/True positive rate (召回率/敏感性)
- **F1-Score**: Harmonic mean of precision and recall (精确率和召回率的调和平均数)

## Deployment Pipeline Flow / 部署流水线流程

### Code to Production Flow / 代码到生产流程

```
Commit → Build → Test → Staging → Production
提交 → 构建 → 测试 → 预发布 → 生产
```

### Branch Triggers / 分支触发器

- **Feature branches**: CI only (功能分支：仅CI)
- **dev branch**: CI + staging deployment (dev分支：CI + 预发布部署)
- **staging branch**: CI + staging deployment (staging分支：CI + 预发布部署)
- **main branch**: CI + production deployment (main分支：CI + 生产部署)

## Contributing / 贡献

1. Create a feature branch from `dev` (从`dev`创建功能分支)
2. Implement your changes (实现你的更改)
3. Add comprehensive tests (添加全面测试)
4. Ensure all CI checks pass (确保所有CI检查通过)
5. Create a pull request to `dev` (创建到`dev`的拉取请求)
6. After review, merge to `dev` (审查后，合并到`dev`)
7. Deploy to staging for testing (部署到预发布进行测试)
8. Merge `dev` to `staging` for staging deployment (将`dev`合并到`staging`进行预发布部署)
9. Merge `staging` to `main` for production deployment (将`staging`合并到`main`进行生产部署)

## License / 许可证

This project is licensed under the MIT License.

本项目采用MIT许可证。