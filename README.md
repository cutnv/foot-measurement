# Foot Measurement

基于 Flask、OpenCV 和轻量模型的三视角足部尺寸测量网页。用户将裸足踩在 A4 纸上，上传正上方、左侧斜拍、右侧斜拍三张照片，系统输出脚长、脚掌宽、脚跟宽、置信度和轮廓结果图。

> 当前为开发版本。测量精度依赖照片清晰度、光线、A4 纸边可见程度和拍摄角度，暂不能替代专业量脚设备。

## 在线使用

已部署的测试网页：<https://47.114.92.29/>

下面出现的 `localhost:5000` 仅用于访问当前电脑上自行启动的服务，不是线上服务器地址。
公网只开放测量页面；数据后台使用 SSH 隧道访问，不开放公网入口。

## 快速启动（Docker）

需要安装 Docker Desktop，并确保 Docker 已启动。

```bash
git clone https://github.com/cutnv/foot-measurement.git
cd foot-measurement
docker compose -f web/docker-compose.yml up -d --build
```

浏览器打开：<http://localhost:5000>

默认会同时启动网页和本地 PostgreSQL。正式使用前请复制
`web/.env.example` 为自己的环境变量文件，替换其中所有密码和
`SAVE_TOKEN_SECRET`，再通过 `--env-file web/.env` 启动。

停止服务：

```bash
docker compose -f web/docker-compose.yml down
```

## 本地 Python 启动

需要 Python 3.11 或更高版本。

Windows：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r web\requirements.txt
.venv\Scripts\python web\app.py
```

Linux / macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r web/requirements.txt
.venv/bin/python web/app.py
```

然后打开：<http://localhost:5000>

## 拍摄要求

1. 裸足自然承重，脚跟最后端凸起点贴齐 A4 纸底部短边。
2. 第一张从正上方拍摄，镜头尽量与纸面平行。
3. 第二、三张分别从脚的左右两侧约 25°—40° 拍摄，推荐 30°—35°。
4. 三张照片使用同一手机、同一 1× 倍率；拍摄期间脚和 A4 纸保持不动。
5. 保证脚尖、脚侧和至少三条纸边清晰可见，避免模糊、强阴影、彩色灯光和 0.5× 超广角。

## 项目结构

```text
models/                         当前网页使用的三个模型
web/app.py                      Flask 接口和测量算法
web/templates/index.html        网页界面
web/static/guide/               拍摄示例图
web/test_*.py                   测试脚本
web/evaluate_*.py               评估脚本
web/requirements.txt            Python 依赖
web/Dockerfile                  应用镜像
web/docker-compose.yml          本地 Docker 启动配置
web/nginx-foot.conf             Nginx 反向代理示例
db/migrations/                  PostgreSQL 版本化结构迁移
db/init/                        数据库最小权限账号初始化
```

## 测试

基础语法和纸张恢复测试：

```bash
python -m py_compile web/app.py
python web/test_paper_recovery.py
```

部分批量回归测试依赖未上传的本地照片数据集，因此无法在全新克隆的仓库中直接运行。

## 数据与部署说明

- 仓库不包含真实用户照片、训练数据集、虚拟环境、缓存和调试产物。
- 用户开始测量前会看到数据说明；测量成功后系统自动保存匿名编号、足别、尺寸、置信度、提示、轮廓图和时间，不保存三张原照片。
- 保存凭证由服务器签名且 30 分钟过期；保存成功或过期后会清理临时结果图。
- PostgreSQL 数据保留两年，`pg_cron` 每日清理过期记录。
- 线上数据库和应用端口仅绑定 `127.0.0.1`；网页账号只能调用保存函数，维护账号只读。
- 开发阶段尚未配置备份。正式应用前必须增加备份、恢复演练和监控。
- 当前测试服务器已启用 HTTPS，并在 Nginx 层拒绝公网 `/admin` 请求。
- Nginx 配置文件为 SSH 私有后台示例；正式部署时仍需按实际 IP、域名和证书修改。

## 数据后台

后台地址：`/admin`。支持按测量编号查询、分页查看、导出 CSV，以及查看或下载轮廓图；后台不提供修改和删除数据的功能。

线上后台不直接暴露公网。维护人员先建立 SSH 隧道：

```powershell
ssh -N -L 15000:127.0.0.1:5000 SSH账号@服务器地址
```

保持终端开启，再访问 <http://127.0.0.1:15000/admin/login>。每位维护人员应使用独立 SSH 账号和密钥；不要共享服务器私钥，也不要把后台密码提交到仓库。

后台使用独立只读数据库账号。启动前配置：

```dotenv
ADMIN_DATABASE_URL=postgresql://foot_reader:密码@数据库地址:5432/foot_measurement
ADMIN_USERNAME=管理员账号
ADMIN_PASSWORD=管理员密码
ADMIN_SESSION_SECRET=至少32位随机字符串
```

本地 HTTP 或 SSH 隧道访问时使用 `SESSION_COOKIE_SECURE=false`；改为域名 HTTPS 后使用 `true`。未配置管理员账号、密码或只读数据库连接时，后台保持禁用。

## 当前使用的模型

- `paper_candidate_ranker.npz`
- `paper_lraspp_amodal_v1.onnx`
- `universal_measurement_v2.npz`

## 许可

本项目目前未声明开源许可证。未经仓库所有者许可，不得复制、分发或商用。
