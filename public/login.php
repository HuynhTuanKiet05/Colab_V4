<?php
require_once __DIR__ . '/../app/services/AuthService.php';

if (is_logged_in()) {
    redirect('index.php');
}

$error = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim($_POST['username'] ?? '');
    $password = trim($_POST['password'] ?? '');

    if (AuthService::attemptLogin($username, $password)) {
        redirect('index.php');
    }

    $error = 'Sai tên đăng nhập hoặc mật khẩu.';
}
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Đăng nhập · AMNTDDA AI</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="login-wrap">
    <div class="login-box">
        <div class="spacer-lg">
            <div class="brand brand-inline">AMNTDDA</div>
            <h1>Đăng nhập hệ thống</h1>
            <p class="muted">Nền tảng AI chẩn đoán liên kết thuốc – bệnh trên đồ thị dị thể (HGT) với giao diện sáng, thống nhất và dễ sử dụng.</p>
        </div>

        <?php if ($error): ?>
            <div class="alert alert-error"><?= e($error) ?></div>
        <?php endif; ?>

        <form method="post">
            <div class="spacer-md">
                <label>Tên tài khoản</label>
                <input class="input" type="text" name="username" placeholder="Nhập tên đăng nhập" required>
            </div>

            <div class="spacer-lg">
                <label>Mật khẩu</label>
                <input class="input" type="password" name="password" placeholder="Nhập mật khẩu" required>
            </div>

            <button class="btn btn-full" type="submit">Đăng nhập</button>
        </form>

        <div class="login-panel-note">
            <p class="muted">Tài khoản mặc định: <strong>admin / password</strong></p>
        </div>

        <div class="feature-list">
            <div class="feature-item"><span class="feature-dot"></span><span>Chẩn đoán liên kết thuốc – bệnh theo Top-K trong vài giây.</span></div>
            <div class="feature-item"><span class="feature-dot"></span><span>So sánh trực tiếp mô hình gốc và mô hình cải tiến trên cùng bộ dữ liệu.</span></div>
            <div class="feature-item"><span class="feature-dot"></span><span>Lưu lịch sử truy vấn và khu vực quản trị thống nhất.</span></div>
        </div>
    </div>
</div>
</body>
</html>
