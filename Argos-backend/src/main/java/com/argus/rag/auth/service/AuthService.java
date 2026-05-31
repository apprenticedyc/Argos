package com.argus.rag.auth.service;

import com.argus.rag.auth.model.dto.RegisterRequest;
import com.argus.rag.auth.security.AccessTokenService;
import com.argus.rag.auth.security.RefreshTokenService;
import com.argus.rag.common.enums.SystemRole;
import com.argus.rag.common.enums.UserStatus;
import com.argus.rag.common.exception.BusinessException;
import com.argus.rag.user.mapper.UserMapper;
import com.argus.rag.user.model.entity.User;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.conditions.update.LambdaUpdateWrapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Clock;
import java.time.LocalDateTime;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * 认证服务，处理登录、注册、令牌刷新、登出等核心逻辑。
 */
@Slf4j
@Service
public class AuthService {

    private static final String INVALID_CREDENTIALS_MESSAGE = "账号或密码错误";
    private static final int MAX_LOGIN_ID_LENGTH = 128;
    private static final int MAX_USERNAME_LENGTH = 64;
    private static final int MAX_EMAIL_LENGTH = 128;
    private static final int MAX_DISPLAY_NAME_LENGTH = 128;
    private static final int MAX_PASSWORD_LENGTH = 256;
    /** BCrypt 最大输入字节数，超出会截断 */
    private static final int BCRYPT_MAX_PASSWORD_BYTES = 72;
    /** 用户名仅允许字母、数字、下划线、短横线 */
    private static final Pattern USERNAME_PATTERN = Pattern.compile("^[A-Za-z0-9_-]+$");
    /** 保留用户名，防止注册时冒充系统账号 */
    private static final Set<String> RESERVED_USERNAMES = Set.of("admin", "root", "null", "undefined", "system");

    private final UserMapper userMapper;
    private final PasswordHasher passwordHasher;
    private final PasswordPolicyValidator passwordValidator;
    private final AccessTokenService accessTokenService;
    private final RefreshTokenService refreshTokenService;
    private final Clock clock;

    public AuthService(
            UserMapper userMapper,
            PasswordHasher passwordHasher,
            PasswordPolicyValidator passwordValidator,
            AccessTokenService accessTokenService,
            RefreshTokenService refreshTokenService,
            Clock clock
    ) {
        this.userMapper = userMapper;
        this.passwordHasher = passwordHasher;
        this.passwordValidator = passwordValidator;
        this.accessTokenService = accessTokenService;
        this.refreshTokenService = refreshTokenService;
        this.clock = clock;
    }

    /**
     * 登录：验证账号密码，吊销旧 refresh token，签发新 access token 和 refresh token。
     */
    @Transactional
    public AuthTokens login(String loginId, String password) {
        loginId = normalizeLoginId(loginId);   // 去空格、长度校验
        validatePassword(password);            // 密码非空 + 长度校验
        User user = loadUserForLogin(loginId); // 按用户名/邮箱查用户（FOR UPDATE 行锁）
        checkPassword(user, password);         // 校验明文密码BCrypt哈希后是否匹配
        refreshTokenService.revokeAllActiveTokens(user.getId()); // 吊销旧 refresh token（单设备互斥）
        RefreshTokenService.RefreshToken refreshToken = refreshTokenService.grantRefreshToken(user.getId()); // 签发新 refresh token
        updateSuccessfulLogin(user.getId());   // 更新最后登录时间
        log.info("用户登录成功: userId={}, username={}", user.getId(), user.getUsername());
        // 登录成功后返回 access token、refresh token 和当前用户信息
        return new AuthTokens(
                user.getId(),
                issueAccessToken(user),        // 签发 JWT access token
                refreshToken.refreshToken(),
                Boolean.TRUE.equals(user.getMustChangePassword()),
                toCurrentUser(user)
        );
    }

    /**
     * 注册：校验用户名/邮箱唯一性，创建用户。
     */
    @Transactional
    public void register(RegisterRequest request) {
        if (request == null) {
            throw new BusinessException("注册请求不能为空");
        }
        String username = normalizeUsername(request.username());
        String email = normalizeRequiredValue(request.email(), "邮箱不能为空", "邮箱长度不能超过 128", MAX_EMAIL_LENGTH);
        String displayName = normalizeRequiredValue(
                request.displayName(), "显示名称不能为空", "显示名称长度不能超过 128", MAX_DISPLAY_NAME_LENGTH);
        passwordValidator.validate(request.password());
        ensureUniqueIdentity(username, email);
        User user = new User();
        user.setUserCode(username);
        user.setUsername(username);
        user.setEmail(email);
        user.setDisplayName(displayName);
        user.setPasswordHash(passwordHasher.hash(request.password()));
        user.setSystemRole(SystemRole.USER);
        user.setStatus(UserStatus.ACTIVE);
        user.setMustChangePassword(false);
        userMapper.insert(user);
        log.info("用户注册成功: username={}, email={}", user.getUsername(), user.getEmail());
    }

    /**
     * 刷新令牌：验证 refresh token，原子吊销旧的，签发新的。
     */
    @Transactional
    public AuthTokens refresh(String refreshToken) {
        if (refreshToken == null || refreshToken.isBlank()) {
            throw new BusinessException("refresh token 不存在或已失效");
        }
        // 解析 token 格式，BCrypt 比对 + 检查未吊销未过期
        RefreshTokenRecord activeToken = refreshTokenService.resolveRefreshToken(refreshToken)
                .orElseThrow(() -> new BusinessException("refresh token 不存在或已失效"));
        User user = userMapper.selectById(activeToken.userId());
        if (user == null) {
            throw new BusinessException("用户不存在");
        }
        ensureRefreshAllowed(user); // 校验账号未被禁用
        // 原子吊销旧 token（乐观锁：WHERE revoked_at IS NULL），失败说明已被抢先使用
        if (!refreshTokenService.revokeTokenById(activeToken.id())) {
            refreshTokenService.revokeAllActiveTokens(user.getId()); // 吊销该用户全部 token
            log.warn("refresh token 重放攻击: userId={}", user.getId());
            throw new BusinessException("refresh token 已被使用，请重新登录");
        }
        RefreshTokenService.RefreshToken nextRefreshToken = refreshTokenService.grantRefreshToken(user.getId()); // 签发新 refresh token
        log.info("令牌刷新成功: userId={}", user.getId());
        return new AuthTokens(
                user.getId(),
                issueAccessToken(user),        // 签发新 JWT access token
                nextRefreshToken.refreshToken(),
                Boolean.TRUE.equals(user.getMustChangePassword()),
                toCurrentUser(user)
        );
    }

    /**
     * 登出：吊销当前 refresh token。
     */
    @Transactional
    public void logout(String refreshToken) {
        if (refreshToken == null || refreshToken.isBlank()) {
            return;
        }
        boolean revoked = refreshTokenService.revokeToken(refreshToken);
        if (revoked) {
            log.info("用户登出成功");
        }
    }

    /** 将 User 实体转换为 CurrentUser，避免重复查库 */
    private CurrentUserService.CurrentUser toCurrentUser(User user) {
        return new CurrentUserService.CurrentUser(
                user.getId(),
                user.getUserCode(),
                user.getDisplayName(),
                user.getSystemRole(),
                Boolean.TRUE.equals(user.getMustChangePassword())
        );
    }

    /** 校验密码非空且长度合法 */
    private void validatePassword(String password) {
        if (password == null || password.isBlank()) {
            throw new BusinessException("密码不能为空");
        }
        if (password.length() > MAX_PASSWORD_LENGTH) {
            throw new BusinessException("密码长度非法");
        }
        if (password.getBytes(StandardCharsets.UTF_8).length > BCRYPT_MAX_PASSWORD_BYTES) {
            throw new BusinessException("密码长度超过安全上限，请控制在 72 字节以内");
        }
    }

    /** 登录标识归一化：去空格、长度校验 */
    private String normalizeLoginId(String loginId) {
        if (loginId == null || loginId.isBlank()) {
            throw new BusinessException("登录标识不能为空");
        }
        String normalizedLoginId = loginId.trim();
        if (normalizedLoginId.length() > MAX_LOGIN_ID_LENGTH) {
            throw new BusinessException("登录标识长度非法");
        }
        return normalizedLoginId;
    }

    /** 用户名归一化并校验 */
    private String normalizeUsername(String username) {
        String normalizedValue = normalizeRequiredValue(username, "用户名不能为空", "用户名长度不能超过 64", MAX_USERNAME_LENGTH);
        if (!USERNAME_PATTERN.matcher(normalizedValue).matches()) {
            throw new BusinessException("用户名不合法");
        }
        if (RESERVED_USERNAMES.contains(normalizedValue.toLowerCase(Locale.ROOT))) {
            throw new BusinessException("用户名不合法");
        }
        return normalizedValue;
    }

    /** 通用归一化方法：去空格 + 长度校验 */
    private String normalizeRequiredValue(String value, String blankMessage, String lengthMessage, int maxLength) {
        if (value == null || value.isBlank()) {
            throw new BusinessException(blankMessage);
        }
        String normalizedValue = value.trim();
        if (normalizedValue.length() > maxLength) {
            throw new BusinessException(lengthMessage);
        }
        return normalizedValue;
    }

    /** 确保用户名和邮箱未被占用 */
    private void ensureUniqueIdentity(String username, String email) {
        if (userMapper.exists(new LambdaQueryWrapper<User>().eq(User::getUsername, username))) {
            throw new BusinessException("用户名已存在");
        }
        if (userMapper.exists(new LambdaQueryWrapper<User>().eq(User::getEmail, email))) {
            throw new BusinessException("邮箱已存在");
        }
    }

    /** 按登录标识查找用户（支持用户名或邮箱），加行锁防止并发 */
    private User loadUserForLogin(String loginId) {
        List<User> users = userMapper.selectByLoginIdForUpdate(loginId);
        if (users.isEmpty()) {
            throw new BusinessException(INVALID_CREDENTIALS_MESSAGE);
        }
        ensureUniqueLoginMatch(users);
        return users.getFirst();
    }

    /** 确保一个登录标识只匹配到一个用户 */
    private void ensureUniqueLoginMatch(List<User> users) {
        Set<Long> userIds = new LinkedHashSet<>();
        for (User user : users) {
            userIds.add(user.getId());
        }
        if (userIds.size() > 1) {
            throw new BusinessException("登录标识存在冲突，请联系管理员处理");
        }
    }

    /** 校验用户是否可以登录：状态正常 + 密码匹配 */
    private void checkPassword(User user, String password) {
        if (user.getStatus() == UserStatus.DISABLED) {
            throw new BusinessException("账号已被禁用");
        }
        if (user.getPasswordHash() == null || !passwordHasher.matches(password, user.getPasswordHash())) {
            throw new BusinessException(INVALID_CREDENTIALS_MESSAGE);
        }
    }

    /** 刷新令牌前校验用户状态 */
    private void ensureRefreshAllowed(User user) {
        if (user.getStatus() == UserStatus.DISABLED) {
            throw new BusinessException("账号已被禁用");
        }
    }

    /** 登录成功后更新最后登录时间，使用 LambdaUpdateWrapper 精确更新单个字段 */
    private void updateSuccessfulLogin(Long userId) {
        LambdaUpdateWrapper<User> updateWrapper = new LambdaUpdateWrapper<>();
        updateWrapper.eq(User::getId, userId)
                      .set(User::getLastLoginAt, LocalDateTime.now(clock));
        userMapper.update(null, updateWrapper);
    }

    /** 签发 access token（JWT） */
    private String issueAccessToken(User user) {
        return accessTokenService.issueToken(
                new AccessTokenService.TokenSubject(
                        user.getId(),
                        user.getUserCode(),
                        user.getDisplayName(),
                        user.getSystemRole(),
                        Boolean.TRUE.equals(user.getMustChangePassword())
                )
        );
    }

    /** 登录/刷新后的令牌集合，包含当前用户信息避免调用方重复查库 */
    public record AuthTokens(
            Long userId,
            String accessToken,
            String refreshToken,
            boolean mustChangePassword,
            CurrentUserService.CurrentUser currentUser
    ) {
    }
}
