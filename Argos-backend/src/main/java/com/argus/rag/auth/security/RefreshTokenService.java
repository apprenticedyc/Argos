package com.argus.rag.auth.security;

import com.argus.rag.auth.config.AuthProperties;
import com.argus.rag.auth.mapper.UserRefreshTokenMapper;
import com.argus.rag.auth.model.entity.UserRefreshToken;
import com.argus.rag.auth.service.PasswordHasher;
import com.argus.rag.auth.service.RefreshTokenRecord;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import org.springframework.stereotype.Service;

import java.security.SecureRandom;
import java.time.Clock;
import java.time.LocalDateTime;
import java.util.Base64;
import java.util.Optional;
import java.util.UUID;

/**
 * Refresh Token 管理服务。
 * <p>
 * Refresh token 格式为 {@code tokenId.secret}，tokenId 明文字段索引查找记录，
 * secret 部分 bcrypt 哈希存储。签发和吊销均操作 user_refresh_tokens 表。
 */
@Service
public class RefreshTokenService {

    private static final String TOKEN_SEPARATOR = ".";
    /** 随机 secret 字节数，Base64URL 编码后约 32 字符 */
    private static final int SECRET_BYTES = 24;

    private final UserRefreshTokenMapper tokenMapper;
    private final PasswordHasher passwordHasher;
    private final AuthProperties authProperties;
    private final Clock clock;
    private final SecureRandom secureRandom = new SecureRandom();

    public RefreshTokenService(UserRefreshTokenMapper tokenMapper, PasswordHasher passwordHasher,
                               AuthProperties authProperties, Clock clock) {
        this.tokenMapper = tokenMapper;
        this.passwordHasher = passwordHasher;
        this.authProperties = authProperties;
        this.clock = clock;
    }

    /**
     * 签发新的 refresh token
     * - 生成随机 secret 作为 token 一部分，bcrypt 哈希后存储
     * - tokenId 明文字段索引查找记录，secret 部分 bcrypt 哈
     * - 返回完整 token 给调用方，供后续验证使用
     */
    public RefreshToken grantRefreshToken(Long userId) {
        LocalDateTime now = LocalDateTime.now(clock);
        // 生成全局唯一令牌ID
        String tokenId = UUID.randomUUID().toString().replace("-", "");
        // 拼接完整明文刷新令牌：唯一ID + 分隔符 + 安全随机密钥
        String refreshToken = tokenId + TOKEN_SEPARATOR + newTokenSecretKey();
        // 计算刷新令牌过期时间
        LocalDateTime expiresAt = now.plusDays(authProperties.getRefreshTokenExpireDays());

        // 构建数据库存储实体
        UserRefreshToken entity = new UserRefreshToken();
        entity.setUserId(userId);
        entity.setTokenId(tokenId);
        // 明文Token哈希加密后入库，不存储明文
        entity.setTokenHash(passwordHasher.hash(refreshToken));
        entity.setExpiresAt(expiresAt);
        entity.setCreatedAt(now);
        // 持久化到数据库
        tokenMapper.insert(entity);

        // 返回明文刷新令牌及令牌记录
        return new RefreshToken(refreshToken, toRecord(entity));
    }

    /** 验证并解析 refresh token，返回对应的结构化实体类
     */
    public Optional<RefreshTokenRecord> resolveRefreshToken(String refreshToken) {
        Optional<ParsedRefreshToken> parsedToken = parseToken(refreshToken);
        if (parsedToken.isEmpty()) {
            return Optional.empty();
        }
        // 根据 tokenId 查找数据库记录
        UserRefreshToken entity = tokenMapper.selectOne(new LambdaQueryWrapper<UserRefreshToken>().eq(UserRefreshToken::getTokenId, parsedToken.get()
                .tokenId()));
        if (entity == null) {
            return Optional.empty();
        }
        RefreshTokenRecord record = toRecord(entity);
        // 校验明文与密文是否匹配
        if (!passwordHasher.matches(refreshToken, record.tokenHash())) {
            return Optional.empty();
        }
        if (!record.isActive(LocalDateTime.now(clock))) {
            return Optional.empty();
        }
        return Optional.of(record);
    }

    /** 吊销指定用户所有有效 token（登录时调用，实现单设备登录） */
    public void revokeAllActiveTokens(Long userId) {
        LocalDateTime now = LocalDateTime.now(clock);
        UserRefreshToken update = new UserRefreshToken();
        update.setRevokedAt(now);
        // 查询用户所有未过期且为吊销的token, 批量更新 revoked_at 字段为当前时间
        tokenMapper.update(update, new LambdaQueryWrapper<UserRefreshToken>().eq(UserRefreshToken::getUserId, userId)
                .isNull(UserRefreshToken::getRevokedAt).gt(UserRefreshToken::getExpiresAt, now));
    }

    /** 吊销单条 token
     *
     */
    public boolean revokeToken(String refreshToken) {
        Optional<RefreshTokenRecord> activeToken = resolveRefreshToken(refreshToken);
        if (activeToken.isEmpty()) {
            return false;
        }
        return revokeTokenById(activeToken.get().id());
    }

    /**
     * 原子吊销指定 ID 的 token：仅当 revoked_at 为 null 时才更新
     * 返回 true 表示成功吊销，false 表示已被其他请求抢先吊销（疑似 token 窃取）。
     */
    public boolean revokeTokenById(Long id) {
        LocalDateTime now = LocalDateTime.now(clock);
        return tokenMapper.revokeByIdIfActive(id, now) > 0;
    }

    /** 解析 token 格式：tokenId.secret */
    private Optional<ParsedRefreshToken> parseToken(String refreshToken) {
        if (refreshToken == null || refreshToken.isBlank()) {
            return Optional.empty();
        }
        String[] segments = refreshToken.trim().split("\\Q" + TOKEN_SEPARATOR + "\\E", 2);
        if (segments.length != 2 || segments[0].isBlank() || segments[1].isBlank()) {
            return Optional.empty();
        }
        return Optional.of(new ParsedRefreshToken(segments[0], segments[1]));
    }

    /** 生成安全随机秘钥作为token一部分 */
    private String newTokenSecretKey() {
        byte[] bytes = new byte[SECRET_BYTES];
        secureRandom.nextBytes(bytes);
        return Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
    }

    private static RefreshTokenRecord toRecord(UserRefreshToken entity) {
        return new RefreshTokenRecord(entity.getId(), entity.getUserId(), entity.getTokenId(), entity.getTokenHash(), entity.getExpiresAt(), entity.getRevokedAt());
    }

    private record ParsedRefreshToken(String tokenId, String secret) {
    }

    /** 签发后返回给调用方的 token 信息 */
    public record RefreshToken(String refreshToken, RefreshTokenRecord record) {
    }
}
