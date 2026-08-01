use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for a chat message.
pub type MessageId = uuid::Uuid;

/// Identifier for a QQ group.
pub type GroupId = String;

/// Identifier for a QQ user.
pub type UserId = String;

/// A raw chat message captured from the QQ group.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub id: MessageId,
    pub group_id: GroupId,
    pub sender: UserId,
    pub nickname: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub reply_to: Option<MessageId>,
    /// Original payload from the WebSocket frame, kept for diagnostics.
    pub raw_payload: serde_json::Value,
}

impl ChatMessage {
    pub fn word_count(&self) -> usize {
        // Simple heuristic: count whitespace-separated tokens.
        self.content.split_whitespace().count()
    }

    pub fn char_count(&self) -> usize {
        self.content.chars().count()
    }
}

/// Affective state maintained per (group, user).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct AffectiveState {
    /// Energy level; increases with activity, decays over time.
    pub energy: f32,
    /// Favorability toward the user; moves based on interaction quality.
    pub favorability: f32,
    /// Number of times the bot has replied to this user.
    pub reply_count: u64,
    /// Last time this state was updated.
    pub last_updated: Option<DateTime<Utc>>,
}

impl AffectiveState {
    /// Apply a reply interaction.
    pub fn on_reply(&mut self, quality: f32, now: DateTime<Utc>) {
        self.reply_count += 1;
        self.energy = (self.energy + 0.1).min(1.0);
        self.favorability = (self.favorability + quality).clamp(-1.0, 1.0);
        self.last_updated = Some(now);
    }

    /// Apply decay based on elapsed minutes.
    pub fn decay(&mut self, minutes: f32, energy_decay_rate: f32, favor_decay_rate: f32) {
        self.energy = (self.energy - minutes * energy_decay_rate).max(0.0);
        let favor_abs = self.favorability.abs();
        let new_abs = (favor_abs - minutes * favor_decay_rate).max(0.0);
        self.favorability = self.favorability.signum() * new_abs;
    }
}

/// Events flow between layers through bounded async channels.
#[derive(Debug, Clone)]
pub enum Event {
    /// A new message arrived from the ingress.
    IncomingMessage(ChatMessage),
    /// Affective state for a user changed.
    AffectiveUpdated(GroupId, UserId, AffectiveState),
    /// Layer 2 produced a decision.
    GateDecision(GateDecision),
    /// Layer 2 decided this message needs council attention.
    Escalation(EscalationReason, ChatMessage, AffectiveState),
    /// Layer 3 output.
    CouncilDecision(CouncilDecision),
    /// Layer 4 input.
    DetectiveRequest(DetectiveInput),
    /// Layer 4 output.
    DetectiveReport(DetectiveReport),
    /// A reply should be sent out.
    ReplyOut(ReplyOut),
    /// Daily solidification trigger.
    SolidifyTick,
    /// Nightly graph summary produced by solidify, fed back to the council so
    /// long-term relationships influence future replies.
    DailyContext(String),
}

/// Reasons for dropping or escalating a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DropReason {
    TooLong,
    Spam,
    BannedUser,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EscalationReason {
    Conflict,
    StaleMeme,
    MaliciousContent,
    NeedsContext,
    Other(String),
}

/// Decision made by the cheap-model gate.
#[derive(Debug, Clone, PartialEq)]
pub enum GateDecision {
    /// Drop the message without reply.
    Drop(DropReason),
    /// Reply with a short cozy sentence (<= 10 Chinese words / tokens).
    CozyReply(String),
    /// Escalate to the council layer.
    Escalate(EscalationReason),
}

/// Input to the council layer.
#[derive(Debug, Clone)]
pub struct CouncilInput {
    pub message: ChatMessage,
    pub state: AffectiveState,
    pub context: Vec<ChatMessage>,
    /// Nightly graph summary (from solidify), empty when none has been
    /// produced yet.
    pub daily_context: String,
}

/// Decision made by the council layer.
#[derive(Debug, Clone)]
pub struct CouncilDecision {
    pub action: CouncilAction,
    pub reasoning: String,
    pub draft_reply: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouncilAction {
    ReplyDirectly,
    LaunchDetective,
    Ignore,
}

/// Input to the detective layer.
#[derive(Debug, Clone)]
pub struct DetectiveInput {
    pub message: ChatMessage,
    pub state: AffectiveState,
    pub target_user: UserId,
}

/// A structured, dehydrated report produced by the detective.
///
/// `message_id`/`group_id` are filled in by the detective actor (not the LLM)
/// so the report can be correlated back to the message that triggered it.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DetectiveReport {
    /// The message id that triggered this report. Set by the detective actor.
    #[serde(default)]
    pub message_id: MessageId,
    /// The group the target user belongs to. Set by the detective actor.
    #[serde(default)]
    pub group_id: GroupId,
    pub target_user: UserId,
    pub summary: String,
    pub historical_facts: Vec<Fact>,
    pub psychological_weaknesses: Vec<Weakness>,
    pub relationship_changes: Vec<RelationshipChange>,
    pub recommended_tone: Tone,
    pub confidence: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Fact {
    pub text: String,
    pub evidence: Vec<String>,
    pub occurred_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Weakness {
    pub description: String,
    pub severity: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct RelationshipChange {
    pub from: UserId,
    pub to: UserId,
    pub kind: RelationKind,
    pub delta: f32,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelationKind {
    #[default]
    Neutral,
    Tease,
    Support,
    Conflict,
    Intimacy,
    Other(String),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub enum Tone {
    #[default]
    Neutral,
    Warm,
    Cold,
    Playful,
    Sarcastic,
    Cautious,
}

/// A graph update expressed as a Cypher statement with bound parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphUpdate {
    pub cypher: String,
    pub params: HashMap<String, serde_json::Value>,
}

/// A reply message to be sent out.
#[derive(Debug, Clone)]
pub struct ReplyOut {
    pub id: MessageId,
    pub reply_to: MessageId,
    pub group_id: GroupId,
    pub target_user: UserId,
    pub content: String,
    pub layer: String,
}

/// A vector-memory record stored in Qdrant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub id: String,
    pub group_id: GroupId,
    pub speaker_id: UserId,
    pub target_id: Option<UserId>,
    pub text: String,
    pub timestamp: DateTime<Utc>,
    pub relation_delta: Option<RelationshipChange>,
    pub tags: Vec<String>,
    pub layer: String,
}

/// Request sent to an LLM provider.
#[derive(Debug, Clone)]
pub struct LlmRequest {
    pub messages: Vec<LlmMessage>,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: LlmRole,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum LlmRole {
    System,
    User,
    Assistant,
}

/// Expected response format for an LLM call.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
}

/// Response from an LLM provider.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub usage: TokenUsage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    Other(String),
}
