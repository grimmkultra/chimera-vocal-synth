//! MAL LABS experimental, preset-driven English speech renderer.
//! Additive CLI entry point. Does not change CHIMERA's live engine defaults.

use crate::{Ctx, FS, peak, wav_name};
use crate::render::g2p;
use chimera::anatomy::human::BodyMix;
use chimera::phonetics::planner::{PhraseInput, PlanOptions, plan_phrases};
use chimera::phonetics::english::Phrase;
use chimera::phonetics::ipa::{skipped_ipa, tokenize_ipa};
use chimera::phonetics::Segment;
use std::collections::{HashMap, HashSet};
use chimera::voice::human_voice::{HumanVoice, SourcePatch};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MalPreset {
    name: String,
    body_mix: BodyMix,
    f0: f64,
    #[serde(default = "default_rate")]
    rate: f64,
    #[serde(default)]
    source: SourcePatch,
}

fn default_rate() -> f64 { 1.0 }

/// In @script.json mode, the visible line remains ordinary English while problem words can
/// receive a precise IPA pronunciation. Only selected words bypass CHIMERA's English G2P.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MalScript {
    text: String,
    #[serde(default)]
    pronunciations: HashMap<String, String>,
}

/// Group segments using the same per-word boundaries as the English front end.
fn words_in_phrase(phrase: &Phrase) -> Result<Vec<Vec<Segment>>, String> {
    let mut groups = Vec::new();
    let mut group = Vec::new();
    for segment in &phrase.segments {
        group.push(segment.clone());
        if segment.word_end.unwrap_or(false) {
            groups.push(std::mem::take(&mut group));
        }
    }
    if !group.is_empty() || groups.len() != phrase.words.len() {
        return Err(format!(
            "phonetic alignment failed: {} words versus {} segment groups",
            phrase.words.len(), groups.len()
        ));
    }
    Ok(groups)
}

fn trace_phones(label: &str, phrases: &[Phrase]) -> Result<(), String> {
    for (n, phrase) in phrases.iter().enumerate() {
        let groups = words_in_phrase(phrase)?;
        let words: Vec<String> = phrase.words.iter().zip(groups.iter())
            .map(|(word, group)| format!(
                "{word}=[{}]",
                group.iter().map(|s| s.ipa.as_str()).collect::<Vec<_>>().join(" ")
            ))
            .collect();
        println!("MAL PHONEMES {label} phrase {}: {}", n + 1, words.join(" | "));
    }
    Ok(())
}

fn apply_pronunciations(
    phrases: &mut [Phrase],
    overrides: &HashMap<String, String>,
) -> Result<(), String> {
    for (word, ipa) in overrides {
        if word.is_empty() || !word.chars().all(|c| c.is_ascii_lowercase() || c == '-' || c == '\'') {
            return Err(format!("pronunciation key {word:?} must be a lowercase English word"));
        }
        if ipa.trim().is_empty() {
            return Err(format!("pronunciation for {word:?} is empty"));
        }
        let skipped = skipped_ipa(ipa);
        if !skipped.is_empty() {
            return Err(format!(
                "pronunciation for {word:?} contains unsupported IPA: {}",
                skipped.warnings().join("; ")
            ));
        }
        let segments = tokenize_ipa(ipa);
        if segments.is_empty() || segments.iter().any(|s| s.ipa == "_") {
            return Err(format!("pronunciation for {word:?} must be a single spoken word without pauses"));
        }
    }

    let mut used = HashSet::new();
    for phrase in phrases {
        let groups = words_in_phrase(phrase)?;
        let mut segments = Vec::new();
        for (word, orig) in phrase.words.iter().zip(groups) {
            if let Some(ipa) = overrides.get(word) {
                let mut repl = tokenize_ipa(ipa);
                // Exactly one word-ending marker per replacement, even if the IPA contains
                // separators or a copied word end.
                for s in &mut repl { s.word_end = None; }
                repl.last_mut().unwrap().word_end = Some(true);
                segments.extend(repl);
                used.insert(word.clone());
            } else {
                segments.extend(orig);
            }
        }
        phrase.segments = segments;
    }
    for word in overrides.keys() {
        if !used.contains(word) {
            return Err(format!("pronunciation override {word:?} matched no word in the script"));
        }
    }
    Ok(())
}

fn bounded(name: &str, v: f64, min: f64, max: f64) -> Result<(), String> {
    if !v.is_finite() || v < min || v > max {
        return Err(format!("{name} must be between {min} and {max} (got {v})"));
    }
    Ok(())
}

impl MalPreset {
    fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() { return Err("preset name must not be empty".into()); }
        bounded("bodyMix.female", self.body_mix.female, 0.0, 1.0)?;
        bounded("bodyMix.child", self.body_mix.child, 0.0, 1.0)?;
        bounded("f0", self.f0, 120.0, 1600.0)?;
        bounded("rate", self.rate, 0.7, 1.5)?;
        let s = &self.source;
        for (name, value) in [
            ("source.breathiness", s.breathiness),
            ("source.roughness", s.roughness),
            ("source.fry", s.fry),
            ("source.tension", s.tension),
            ("source.brightness", s.brightness),
        ] {
            if let Some(value) = value { bounded(name, value, 0.0, 1.0)?; }
        }
        if let Some(v) = s.pressure { bounded("source.pressure", v, 2.0, 14.0)?; }
        if let Some(v) = s.drift { bounded("source.drift", v, 0.0, 60.0)?; }
        if let Some(v) = s.vibrato_depth { bounded("source.vibratoDepth", v, 0.0, 1.0)?; }
        if let Some(v) = s.vibrato_rate { bounded("source.vibratoRate", v, 0.0, 12.0)?; }
        if let Some(v) = s.glide { bounded("source.glide", v, 0.005, 0.5)?; }
        if s.f0.is_some() { return Err("use top-level f0 instead of source.f0".into()); }
        Ok(())
    }
}

/// Usage: chimera render mal <preset.json> <output-stem> "English dialogue."
///    or: chimera render mal <preset.json> <output-stem> @scripts/spoken.json
pub fn render(ctx: &Ctx, args: &[String]) -> Result<(), String> {
    if args.len() != 3 {
        return Err("usage: chimera render mal <preset.json> <out-stem> \"English dialogue.\" or @script.json".into());
    }
    let preset_path = &args[0];
    let file = wav_name(&args[1]);
    let (text, overrides) = if let Some(path) = args[2].strip_prefix('@') {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("failed reading spoken script {path:?}: {e}"))?;
        let script: MalScript = serde_json::from_str(&raw)
            .map_err(|e| format!("invalid spoken script {path:?}: {e}"))?;
        (script.text, script.pronunciations)
    } else {
        (args[2].clone(), HashMap::new())
    };
    let text = text.trim();
    if text.is_empty() || text.chars().count() > 500 {
        return Err("dialogue must contain 1 to 500 characters".into());
    }

    let raw = std::fs::read_to_string(preset_path)
        .map_err(|e| format!("failed reading preset {preset_path:?}: {e}"))?;
    let preset: MalPreset = serde_json::from_str(&raw)
        .map_err(|e| format!("invalid preset {preset_path:?}: {e}"))?;
    preset.validate()?;

    let g = g2p()?;
    let mut phrases = g.to_phrases(text);
    if phrases.is_empty() { return Err("text produced no phonetic phrases".into()); }
    trace_phones("auto", &phrases)?;
    apply_pronunciations(&mut phrases, &overrides)?;
    if !overrides.is_empty() { trace_phones("override", &phrases)?; }
    let inputs: Vec<PhraseInput> = phrases.iter().map(|p| PhraseInput {
        segments: p.segments.clone(),
        intonation: Some(p.intonation),
        pause: Some(p.pause),
    }).collect();
    let plan = plan_phrases(&inputs, &PlanOptions {
        body: preset.body_mix,
        t0: 0.05,
        f0: Some(preset.f0),
        rate: Some(preset.rate),
        ..Default::default()
    });

    let seconds = plan.end + 0.3;
    if !seconds.is_finite() || seconds <= 0.0 || seconds > 60.0 {
        return Err(format!("unexpected duration: {seconds:.2} seconds"));
    }
    let mut voice = HumanVoice::new(FS, preset.body_mix.body(), 2);
    let mut source = preset.source;
    source.f0 = Some(preset.f0);
    voice.set_source(&source);
    voice.schedule(&plan.gestures);

    let mut out = vec![0f32; (seconds * FS).round() as usize];
    for block in out.chunks_mut(128) { voice.render(block); }
    ctx.write_wav(&file, &out)?;
    println!(
        "MAL LABS: {} | {:.2}s | peak {:.3} | {}",
        preset.name, out.len() as f64 / FS, peak(&out), file
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_pronunciation_overrides_only_matching_word() {
        let mut phrases = vec![Phrase {
            segments: vec![
                Segment { ipa: "b".into(), ..Default::default() },
                Segment { ipa: "ɪ".into(), ..Default::default() },
                Segment { ipa: "t".into(), word_end: Some(true), ..Default::default() },
                Segment { ipa: "m".into(), ..Default::default() },
                Segment { ipa: "i".into(), word_end: Some(true), ..Default::default() },
            ],
            intonation: chimera::phonetics::Intonation::Statement,
            pause: 0.3,
            words: vec!["bitty".into(), "me".into()],
        }];
        let mut overrides = HashMap::new();
        overrides.insert("bitty".into(), "ˈbɪɾi".into());
        apply_pronunciations(&mut phrases, &overrides).unwrap();
        let parts = words_in_phrase(&phrases[0]).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].iter().map(|s| s.ipa.as_str()).collect::<Vec<_>>(),
            vec!["b", "ɪ", "ɾ", "i"]);
        assert_eq!(parts[1].iter().map(|s| s.ipa.as_str()).collect::<Vec<_>>(), vec!["m", "i"]);
    }

    #[test]
    fn reject_unsupported_ipa_override() {
        let mut overrides = HashMap::new();
        overrides.insert("bitty".into(), "b???".into());
        let mut phrases = Vec::new();
        assert!(apply_pronunciations(&mut phrases, &overrides).is_err());
    }

    #[test]
    fn parse_and_validate_initial_preset() {
        let raw = include_str!("../../../voices/mal_voice_001.json");
        let preset: MalPreset = serde_json::from_str(raw).unwrap();
        preset.validate().unwrap();
        let weights = preset.body_mix.weights();
        assert_eq!(weights[0].1, 0.0);
        assert!((weights[1].1 - 0.7).abs() < 1e-12);
        assert!((weights[2].1 - 0.3).abs() < 1e-12);
    }
}
