/// Scheduling policy for request ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-Come-First-Served: requests served in arrival order
    FCFS,
    /// Priority-based: requests ordered by priority value (lower = higher priority)
    Priority,
    /// Shortest Job First: prioritize requests with smallest output length
    SJF,
}

impl SchedulingPolicy {
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "fcfs" => Ok(SchedulingPolicy::FCFS),
            "priority" => Ok(SchedulingPolicy::Priority),
            "sjf" => Ok(SchedulingPolicy::SJF),
            _ => Err(format!("Unknown scheduling policy: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_from_str() {
        assert_eq!(
            SchedulingPolicy::from_str("fcfs").unwrap(),
            SchedulingPolicy::FCFS
        );
        assert_eq!(
            SchedulingPolicy::from_str("FCFS").unwrap(),
            SchedulingPolicy::FCFS
        );
        assert_eq!(
            SchedulingPolicy::from_str("priority").unwrap(),
            SchedulingPolicy::Priority
        );
        assert_eq!(
            SchedulingPolicy::from_str("sjf").unwrap(),
            SchedulingPolicy::SJF
        );
        assert_eq!(
            SchedulingPolicy::from_str("SJF").unwrap(),
            SchedulingPolicy::SJF
        );
        assert!(SchedulingPolicy::from_str("unknown").is_err());
    }
}
