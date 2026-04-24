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

    $error = 'Sai ten dang nhap hoac mat khau.';
}
?>
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dang nhap</title>
    <link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="login-wrap">
    <div class="login-box">
        <div class="spacer-lg">
            <div class="brand brand-inline">AMNTDDA</div>
            <h1>Dang nhap he thong</h1>
            <p class="muted">Nen tang du doan lien ket thuoc - benh bang AI HGT voi giao dien sang, thong nhat va de su dung.</p>
        </div>

        <?php if ($error): ?>
            <div class="alert alert-error"><?= e($error) ?></div>
        <?php endif; ?>

        <form method="post">
            <div class="spacer-md">
                <label>Tai khoan</label>
                <input class="input" type="text" name="username" placeholder="Nhap username" required>
            </div>

            <div class="spacer-lg">
                <label>Mat khau</label>
                <input class="input" type="password" name="password" placeholder="Nhap mat khau" required>
            </div>

            <button class="btn btn-full" type="submit">Dang nhap</button>
        </form>

        <div class="login-panel-note">
            <p class="muted">Tai khoan mac dinh: <strong>admin / password</strong></p>
        </div>

        <div class="feature-list">
            <div class="feature-item"><span class="feature-dot"></span><span>Tra cuu lien ket thuoc - benh theo top-k.</span></div>
            <div class="feature-item"><span class="feature-dot"></span><span>Do thi 3D truc quan hoa quan he sinh hoc.</span></div>
            <div class="feature-item"><span class="feature-dot"></span><span>Lich su truy van va khu vuc quan tri thong nhat.</span></div>
        </div>
    </div>
</div>
</body>
</html>
