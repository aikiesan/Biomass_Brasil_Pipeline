-- Run ONCE on the production database (Supabase SQL editor or psql on the VM)
-- by an administrator. Creates two roles:
--   bbp_loader : used by `bbp load` — may create and write tables only in schema bbp
--   pilar_reader: used by the PILAR-2b backend — read-only on schema bbp
-- Replace the passwords before running.

CREATE SCHEMA IF NOT EXISTS bbp;

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bbp_loader') THEN
    CREATE ROLE bbp_loader LOGIN PASSWORD 'CHANGE_ME_loader';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'pilar_reader') THEN
    CREATE ROLE pilar_reader LOGIN PASSWORD 'CHANGE_ME_reader';
  END IF;
END $$;

-- loader owns schema bbp (create tables, views, indexes; nothing elsewhere)
ALTER SCHEMA bbp OWNER TO bbp_loader;
DO $$ BEGIN
  EXECUTE format('GRANT CONNECT ON DATABASE %I TO bbp_loader, pilar_reader', current_database());
END $$;

-- reader: select on everything bbp_loader creates in bbp, now and in the future
GRANT USAGE ON SCHEMA bbp TO pilar_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA bbp TO pilar_reader;
-- (if this line fails with "permission denied", run it again connected AS bbp_loader,
--  without the FOR ROLE clause)
ALTER DEFAULT PRIVILEGES FOR ROLE bbp_loader IN SCHEMA bbp GRANT SELECT ON TABLES TO pilar_reader;

-- If the PILAR-2b backend already connects with another role (e.g. Supabase's
-- `postgres` or `authenticator`), grant that role instead of creating pilar_reader:
--   GRANT USAGE ON SCHEMA bbp TO <role>;
--   GRANT SELECT ON ALL TABLES IN SCHEMA bbp TO <role>;
--   ALTER DEFAULT PRIVILEGES FOR ROLE bbp_loader IN SCHEMA bbp GRANT SELECT ON TABLES TO <role>;
