use neko_core::{
    DetectiveReport, Fact, MemoryDecision, NekoError, RelationKind, RelationshipChange, Tone,
    Weakness,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct DetectiveReportJson {
    target_user: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    historical_facts: Vec<FactJson>,
    #[serde(default)]
    psychological_weaknesses: Vec<WeaknessJson>,
    #[serde(default)]
    relationship_changes: Vec<RelationshipChangeJson>,
    #[serde(default)]
    recommended_tone: String,
    #[serde(default)]
    confidence: f32,
}

#[derive(Debug, Deserialize)]
struct FactJson {
    text: String,
    #[serde(default)]
    evidence: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct WeaknessJson {
    description: String,
    #[serde(default)]
    severity: f32,
}

#[derive(Debug, Deserialize)]
struct RelationshipChangeJson {
    from: String,
    to: String,
    #[serde(default)]
    kind: String,
    #[serde(default)]
    delta: f32,
    #[serde(default)]
    evidence: Vec<String>,
}

pub fn parse_detective_report(raw: &str) -> Result<DetectiveReport, NekoError> {
    let cleaned = neko_llm::extract_json(raw).unwrap_or_else(|| raw.trim().to_string());
    let parsed: DetectiveReportJson = serde_json::from_str(&cleaned)
        .map_err(|e| NekoError::parse(format!("cannot parse detective report: {e}")))?;

    Ok(DetectiveReport {
        target_user: parsed.target_user,
        summary: parsed.summary,
        historical_facts: parsed
            .historical_facts
            .into_iter()
            .map(|f| Fact {
                text: f.text,
                evidence: f.evidence,
                occurred_at: None,
            })
            .collect(),
        psychological_weaknesses: parsed
            .psychological_weaknesses
            .into_iter()
            .map(|w| Weakness {
                description: w.description,
                severity: w.severity.clamp(0.0, 1.0),
            })
            .collect(),
        relationship_changes: parsed
            .relationship_changes
            .into_iter()
            .map(|r| RelationshipChange {
                from: r.from,
                to: r.to,
                kind: parse_relation_kind(&r.kind),
                delta: r.delta,
                evidence: r.evidence,
            })
            .collect(),
        recommended_tone: parse_tone(&parsed.recommended_tone),
        confidence: parsed.confidence.clamp(0.0, 1.0),
        // message_id/group_id are filled in by the detective actor from the
        // request; the LLM never supplies them.
        ..Default::default()
    })
}

fn parse_relation_kind(s: &str) -> RelationKind {
    match s.to_ascii_lowercase().as_str() {
        "tease" => RelationKind::Tease,
        "support" => RelationKind::Support,
        "conflict" => RelationKind::Conflict,
        "intimacy" => RelationKind::Intimacy,
        "neutral" => RelationKind::Neutral,
        other => RelationKind::Other(other.to_string()),
    }
}

fn parse_tone(s: &str) -> Tone {
    match s.to_ascii_lowercase().as_str() {
        "warm" => Tone::Warm,
        "cold" => Tone::Cold,
        "playful" => Tone::Playful,
        "sarcastic" => Tone::Sarcastic,
        "cautious" => Tone::Cautious,
        _ => Tone::Neutral,
    }
}

pub fn parse_memory_decision(raw: &str) -> Result<MemoryDecision, NekoError> {
    let cleaned = neko_llm::extract_json(raw).unwrap_or_else(|| raw.trim().to_string());
    serde_json::from_str::<MemoryDecision>(&cleaned)
        .map_err(|e| NekoError::parse(format!("cannot parse memory decision: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use neko_core::{RelationKind, Tone};

    #[test]
    fn parses_full_report() {
        let raw = r#"{
            "target_user": "u1",
            "summary": "summary text",
            "historical_facts": [{"text": "fact1", "evidence": ["e1"]}],
            "psychological_weaknesses": [{"description": "weak", "severity": 1.5}],
            "relationship_changes": [{"from": "a", "to": "b", "kind": "tease", "delta": -0.2, "evidence": ["e2"]}],
            "recommended_tone": "warm",
            "confidence": 2.0
        }"#;
        let report = parse_detective_report(raw).unwrap();
        assert_eq!(report.target_user, "u1");
        assert_eq!(report.summary, "summary text");
        assert_eq!(report.historical_facts.len(), 1);
        assert_eq!(report.psychological_weaknesses[0].severity, 1.0);
        assert_eq!(report.relationship_changes[0].kind, RelationKind::Tease);
        assert_eq!(report.recommended_tone, Tone::Warm);
        assert_eq!(report.confidence, 1.0);
    }

    #[test]
    fn defaults_empty_fields() {
        let raw = r#"{"target_user": "u1", "summary": ""}"#;
        let report = parse_detective_report(raw).unwrap();
        assert!(report.historical_facts.is_empty());
        assert!(report.psychological_weaknesses.is_empty());
        assert!(report.relationship_changes.is_empty());
        assert_eq!(report.recommended_tone, Tone::Neutral);
        assert_eq!(report.confidence, 0.0);
    }

    #[test]
    fn parses_relation_kinds_and_tones() {
        let cases = [
            ("support", RelationKind::Support),
            ("conflict", RelationKind::Conflict),
            ("intimacy", RelationKind::Intimacy),
            ("neutral", RelationKind::Neutral),
            ("custom", RelationKind::Other("custom".to_string())),
        ];
        for (kind_str, expected) in cases {
            let raw = format!(
                r#"{{"target_user":"x","summary":"","relationship_changes":[{{"from":"a","to":"b","kind":"{}","delta":0,"evidence":[]}}]}}"#,
                kind_str
            );
            let report = parse_detective_report(&raw).unwrap();
            assert_eq!(report.relationship_changes[0].kind, expected);
        }

        let raw = r#"{"target_user":"x","summary":"","recommended_tone":"sarcastic"}"#;
        let report = parse_detective_report(raw).unwrap();
        assert_eq!(report.recommended_tone, Tone::Sarcastic);
    }

    #[test]
    fn malformed_json_returns_error() {
        let err = parse_detective_report("not json").unwrap_err();
        assert!(matches!(err, NekoError::Parse(_)));
    }
}
