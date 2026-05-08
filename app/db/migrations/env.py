from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import models so Alembic detects them for autogenerate
from app.models.db import Base  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    # Use synchronous URL for offline mode
    sync_url = url.replace("+asyncpg", "+psycopg2")
    context.configure(
        url=sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # Use synchronous driver for Alembic
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = cfg["sqlalchemy.url"].replace("+asyncpg", "+psycopg2")

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
