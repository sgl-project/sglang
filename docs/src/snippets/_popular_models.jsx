// Auto-rotating carousel over the shared list in
// /src/snippets/configs/popular-models.jsx:
//
//   <PopularModels models={popularModels} variant="hero" />   // docs home
//   <PopularModels models={popularModels} />                  // cookbook home
//
//   variant="hero"   full-width banner per model — headline, blurb, tags, CTA,
//                    brand tile, from the entry's `hero` block.
//   variant="strip"  (default) one compact line per model — brand mark, name,
//                    tags, "Open" button. No prose; the tags carry the pitch.
//
// Rotation pauses on hover/focus and is skipped under prefers-reduced-motion.
//
// One export for both shapes: Mintlify evaluates each exported component on its
// own at hydration, so anything two components would share (rotation state, the
// timer, the control cluster) has to sit inside one — module-level helpers are
// out of scope by the time this runs.

export const PopularModels = ({
  models = [],
  variant = "strip",
  // A hero blurb takes longer to read than one line of tags.
  interval = variant === "hero" ? 9000 : 6000,
  label = "Popular models",
}) => {
  const [index, setIndex] = useState(0);
  const [paused, setPaused] = useState(false);
  const [reduceMotion, setReduceMotion] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduceMotion(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);

  const count = models.length;
  const isHero = variant === "hero";

  // Functional update so the timer never closes over a stale index.
  useEffect(() => {
    if (count < 2 || paused || reduceMotion) return;
    const id = window.setInterval(
      () => setIndex((i) => (i + 1) % count),
      Math.max(2000, interval)
    );
    return () => window.clearInterval(id);
  }, [count, paused, reduceMotion, interval]);

  // A shortened list must not leave the track parked past its last slide.
  const active = count ? Math.min(index, count - 1) : 0;

  if (!count) return null;

  const navButtonStyle = {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    width: "1.15rem",
    height: "1.15rem",
    padding: 0,
    border: "1px solid rgba(255, 255, 255, 0.28)",
    borderRadius: "999px",
    background: "rgba(255, 255, 255, 0.1)",
    color: "rgba(255, 255, 255, 0.92)",
    fontSize: "0.8rem",
    lineHeight: 1,
    cursor: "pointer",
  };

  const controls =
    count > 1 ? (
      <span style={{ display: "inline-flex", alignItems: "center", gap: "0.45rem" }}>
        <button
          type="button"
          onClick={() => setIndex((i) => (i - 1 + count) % count)}
          aria-label="Previous model"
          style={navButtonStyle}
        >
          ‹
        </button>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "0.3rem" }}>
          {models.map((m, i) => (
            <button
              key={m.href || m.name}
              type="button"
              onClick={() => setIndex(i)}
              aria-label={`Show ${m.name}`}
              aria-current={i === active ? "true" : undefined}
              style={{
                width: i === active ? "1.1rem" : "0.4rem",
                height: "0.4rem",
                padding: 0,
                border: 0,
                borderRadius: "999px",
                background:
                  i === active ? "rgba(255, 255, 255, 0.92)" : "rgba(255, 255, 255, 0.34)",
                cursor: "pointer",
                transition: reduceMotion
                  ? "none"
                  : "width 0.25s ease, background 0.25s ease",
              }}
            />
          ))}
        </span>
        <button
          type="button"
          onClick={() => setIndex((i) => (i + 1) % count)}
          aria-label="Next model"
          style={navButtonStyle}
        >
          ›
        </button>
      </span>
    ) : null;

  // Only the active slide is opaque: mid-slide, a visible neighbour reads as two
  // half-drawn cards rather than as motion.
  const slideStyle = (i) => ({
    flex: "0 0 100%",
    minWidth: 0,
    opacity: i === active ? 1 : 0,
    pointerEvents: i === active ? "auto" : "none",
    transition: reduceMotion ? "none" : "opacity 0.3s ease",
  });

  const tagChip = (t, big) => (
    <span
      key={t}
      style={{
        padding: big ? "0.35rem 0.65rem" : "0.15rem 0.45rem",
        borderRadius: "999px",
        background: big ? "rgba(255, 255, 255, 0.1)" : "rgba(255, 255, 255, 0.12)",
        color: "rgba(255, 255, 255, 0.9)",
        fontSize: big ? "0.78rem" : "0.68rem",
        fontWeight: 650,
        whiteSpace: big ? "normal" : "nowrap",
      }}
    >
      {t}
    </span>
  );

  const heroSlide = (m, i) => {
    const hero = m.hero || {};
    const cta = hero.cta || `Open the ${m.name} cookbook`;
    return (
      <div key={m.href || m.name} aria-hidden={i === active ? undefined : "true"} style={slideStyle(i)}>
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            gap: "clamp(1.25rem, 3vw, 2rem)",
          }}
        >
          <div style={{ flex: "1 1 24rem", minWidth: 0 }}>
            <a
              href={m.href}
              tabIndex={i === active ? undefined : -1}
              style={{
                display: "block",
                margin: 0,
                color: "#ffffff",
                fontSize: "clamp(1.75rem, 4vw, 2.65rem)",
                fontWeight: 750,
                lineHeight: 1.08,
                letterSpacing: "-0.035em",
                textDecoration: "none",
              }}
            >
              {hero.headline || m.name}
            </a>
            {hero.blurb ? (
              // A div, not a p: MDX rewrites `p` to its own inline element, so a
              // paragraph here would depend on how that element is styled.
              <div
                style={{
                  maxWidth: "48rem",
                  margin: "1rem 0 0",
                  color: "rgba(255, 255, 255, 0.82)",
                  fontSize: "1rem",
                  lineHeight: 1.65,
                }}
              >
                {hero.blurb}
              </div>
            ) : null}
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "0.5rem",
                marginTop: "1.15rem",
              }}
            >
              {(hero.tags || m.tags || []).map((t) => tagChip(t, true))}
            </div>
            <a
              href={m.href}
              tabIndex={i === active ? undefined : -1}
              style={{
                display: "inline-flex",
                alignItems: "center",
                marginTop: "1.35rem",
                padding: "0.7rem 1rem",
                borderRadius: "0.55rem",
                background: "#ffffff",
                color: "#7c2d12",
                fontSize: "0.88rem",
                fontWeight: 750,
                textDecoration: "none",
              }}
            >
              {cta}&nbsp;→
            </a>
          </div>
          <a
            href={m.href}
            aria-label={cta}
            tabIndex={i === active ? undefined : -1}
            style={{
              flex: "0 1 12rem",
              minWidth: "10rem",
              padding: "0.8rem",
              border: "1px solid rgba(255, 255, 255, 0.22)",
              borderRadius: "0.9rem",
              background: "rgba(255, 255, 255, 0.96)",
              boxShadow: "0 16px 35px rgba(0, 0, 0, 0.22)",
              textDecoration: "none",
            }}
          >
            <div
              role="img"
              aria-label={m.vendor || m.name}
              style={{
                width: "100%",
                aspectRatio: "16 / 9",
                borderRadius: "0.45rem",
                backgroundColor: "#ffffff",
                backgroundImage: `url('${m.logo}')`,
                backgroundPosition: "center",
                backgroundRepeat: "no-repeat",
                backgroundSize: "cover",
              }}
            />
            {hero.caption ? (
              <div
                style={{
                  padding: "0.65rem 0.35rem 0.2rem",
                  color: "#111827",
                  textAlign: "center",
                  fontSize: "0.78rem",
                  fontWeight: 750,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                }}
              >
                {hero.caption}
              </div>
            ) : null}
          </a>
        </div>
      </div>
    );
  };

  const stripSlide = (m, i) => (
    <a
      key={m.href || m.name}
      href={m.href}
      aria-hidden={i === active ? undefined : "true"}
      tabIndex={i === active ? undefined : -1}
      style={{
        ...slideStyle(i),
        display: "flex",
        alignItems: "center",
        gap: "0.75rem",
        color: "#ffffff",
        textDecoration: "none",
      }}
    >
      <span
        role="img"
        aria-label={m.vendor || m.name}
        style={{
          flex: "0 0 auto",
          width: "3.4rem",
          aspectRatio: "16 / 9",
          borderRadius: "0.35rem",
          border: "1px solid rgba(255, 255, 255, 0.22)",
          backgroundColor: "#ffffff",
          backgroundImage: `url('${m.logo}')`,
          backgroundPosition: "center",
          backgroundRepeat: "no-repeat",
          backgroundSize: "cover",
        }}
      />

      <span style={{ flex: "1 1 auto", minWidth: 0 }}>
        <span style={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: "0.4rem" }}>
          <span
            style={{
              fontSize: "1.02rem",
              fontWeight: 750,
              letterSpacing: "-0.02em",
              lineHeight: 1.2,
            }}
          >
            {m.name}
          </span>
          {m.badge ? (
            <span
              style={{
                padding: "0.1rem 0.4rem",
                borderRadius: "999px",
                background: "rgba(255, 255, 255, 0.92)",
                color: "#7c2d12",
                fontSize: "0.6rem",
                fontWeight: 800,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
              }}
            >
              {m.badge}
            </span>
          ) : null}
        </span>

        <span style={{ display: "flex", flexWrap: "wrap", gap: "0.3rem", marginTop: "0.35rem" }}>
          {(m.tags || []).map((t) => tagChip(t, false))}
        </span>
      </span>

      <span
        style={{
          flex: "0 0 auto",
          padding: "0.25rem 0.55rem",
          borderRadius: "0.4rem",
          background: "rgba(255, 255, 255, 0.92)",
          color: "#7c2d12",
          fontSize: "0.7rem",
          fontWeight: 750,
          whiteSpace: "nowrap",
        }}
      >
        Open&nbsp;→
      </span>
    </a>
  );

  return (
    <div className="not-prose">
      <div
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
        onFocus={() => setPaused(true)}
        onBlur={() => setPaused(false)}
        aria-roledescription="carousel"
        aria-label={label}
        style={{
          position: "relative",
          overflow: "hidden",
          margin: isHero ? "0 0 1.5rem" : "1.5rem 0",
          padding: isHero ? "clamp(1.5rem, 4vw, 2.5rem)" : "0.8rem 1rem 0.9rem",
          border: "1px solid rgba(251, 146, 60, 0.35)",
          borderRadius: isHero ? "1rem" : "0.9rem",
          background: "linear-gradient(135deg, #111827 0%, #31202f 58%, #9a3412 100%)",
          boxShadow: isHero
            ? "0 20px 45px rgba(17, 24, 39, 0.18)"
            : "0 14px 32px rgba(17, 24, 39, 0.16)",
          color: "#ffffff",
        }}
      >
        <div
          aria-hidden="true"
          style={{
            position: "absolute",
            top: isHero ? "-7rem" : "-6rem",
            right: isHero ? "-5rem" : "-4rem",
            width: isHero ? "18rem" : "14rem",
            height: isHero ? "18rem" : "14rem",
            borderRadius: "999px",
            background: "rgba(251, 146, 60, 0.18)",
            filter: "blur(2px)",
          }}
        />

        {/* Header line: label on the left, controls on the right. The hero's
            label is the active entry's eyebrow, so it lives here rather than in
            the slide — a floating control cluster would collide with a wide
            eyebrow badge once the card narrows. */}
        <div
          style={{
            position: "relative",
            zIndex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.75rem",
            flexWrap: "wrap",
            marginBottom: isHero ? "0.9rem" : "0.6rem",
          }}
        >
          <span
            style={
              isHero
                ? {
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.45rem",
                    padding: "0.35rem 0.7rem",
                    border: "1px solid rgba(255, 255, 255, 0.28)",
                    borderRadius: "999px",
                    background: "rgba(255, 255, 255, 0.1)",
                    fontSize: "0.72rem",
                    fontWeight: 750,
                    letterSpacing: "0.08em",
                    textTransform: "uppercase",
                  }
                : {
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.35rem",
                    color: "rgba(255, 255, 255, 0.78)",
                    fontSize: "0.66rem",
                    fontWeight: 750,
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                  }
            }
          >
            <span aria-hidden="true">✦</span>
            {isHero
              ? ((models[active] || {}).hero || {}).eyebrow || label
              : label}
          </span>
          {controls}
        </div>

        <div style={{ position: "relative", zIndex: 1, overflow: "hidden" }}>
          <div
            style={{
              display: "flex",
              alignItems: isHero ? "stretch" : "center",
              transform: `translateX(-${active * 100}%)`,
              transition: reduceMotion ? "none" : "transform 0.45s ease",
            }}
          >
            {models.map((m, i) => (isHero ? heroSlide(m, i) : stripSlide(m, i)))}
          </div>
        </div>
      </div>
    </div>
  );
};
