\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS pg_cron;

CREATE SEQUENCE public.shoe_measurement_code_seq
    START WITH 1
    MAXVALUE 999999;

CREATE TABLE public.shoe_measurements (
    measurement_code text PRIMARY KEY DEFAULT
        ('FM-' || lpad(nextval('public.shoe_measurement_code_seq')::text, 6, '0')),
    save_nonce uuid NOT NULL UNIQUE,
    foot_side text NOT NULL CHECK (foot_side IN ('left', 'right')),
    foot_length_mm numeric(5,1) NOT NULL CHECK (foot_length_mm BETWEEN 180 AND 350),
    ball_width_mm numeric(5,1) NOT NULL CHECK (ball_width_mm BETWEEN 60 AND 130),
    heel_width_mm numeric(5,1) CHECK (heel_width_mm BETWEEN 30 AND 100),
    quality_grade text NOT NULL CHECK
        (quality_grade IN ('high', 'medium', 'low', 'unrated')),
    dimension_confidence jsonb NOT NULL DEFAULT '{}'::jsonb CHECK
        (jsonb_typeof(dimension_confidence) = 'object'),
    warnings jsonb NOT NULL DEFAULT '[]'::jsonb CHECK
        (jsonb_typeof(warnings) = 'array'),
    result_image_png bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL DEFAULT (now() + interval '2 years')
);

CREATE INDEX shoe_measurements_expires_at_idx
    ON public.shoe_measurements (expires_at);

CREATE FUNCTION public.lookup_measurement_code(p_save_nonce uuid)
RETURNS text
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT measurement_code
    FROM public.shoe_measurements
    WHERE save_nonce = p_save_nonce
$$;

CREATE FUNCTION public.save_shoe_measurement(
    p_save_nonce uuid,
    p_foot_side text,
    p_foot_length_mm numeric,
    p_ball_width_mm numeric,
    p_heel_width_mm numeric,
    p_quality_grade text,
    p_dimension_confidence jsonb,
    p_warnings jsonb,
    p_result_image_png bytea
)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    saved_code text;
BEGIN
    INSERT INTO public.shoe_measurements (
        save_nonce, foot_side, foot_length_mm, ball_width_mm,
        heel_width_mm, quality_grade, dimension_confidence,
        warnings, result_image_png
    ) VALUES (
        p_save_nonce, p_foot_side, p_foot_length_mm, p_ball_width_mm,
        p_heel_width_mm, p_quality_grade, p_dimension_confidence,
        p_warnings, p_result_image_png
    )
    ON CONFLICT (save_nonce) DO NOTHING
    RETURNING measurement_code INTO saved_code;

    IF saved_code IS NULL THEN
        SELECT measurement_code INTO saved_code
        FROM public.shoe_measurements
        WHERE save_nonce = p_save_nonce;
    END IF;
    RETURN saved_code;
END;
$$;

REVOKE ALL ON TABLE public.shoe_measurements FROM PUBLIC;
REVOKE ALL ON SEQUENCE public.shoe_measurement_code_seq FROM PUBLIC;
REVOKE ALL ON FUNCTION public.lookup_measurement_code(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.save_shoe_measurement(
    uuid, text, numeric, numeric, numeric, text, jsonb, jsonb, bytea
) FROM PUBLIC;

GRANT SELECT ON TABLE public.shoe_measurements TO foot_reader;
GRANT EXECUTE ON FUNCTION public.lookup_measurement_code(uuid) TO foot_app;
GRANT EXECUTE ON FUNCTION public.save_shoe_measurement(
    uuid, text, numeric, numeric, numeric, text, jsonb, jsonb, bytea
) TO foot_app;

SELECT cron.schedule(
    'purge-expired-shoe-measurements',
    '15 3 * * *',
    $$DELETE FROM public.shoe_measurements WHERE expires_at < now()$$
)
WHERE NOT EXISTS (
    SELECT 1 FROM cron.job
    WHERE jobname = 'purge-expired-shoe-measurements'
);
