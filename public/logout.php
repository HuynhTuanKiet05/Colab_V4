<?php
require_once __DIR__ . '/../app/services/AuthService.php';
AuthService::logout();
redirect('login.php');
