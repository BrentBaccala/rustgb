//! Polynomial ring definition.
//!
//! A [`Ring`] bundles the immutable data every polynomial operation needs:
//! number of variables, monomial ordering, and coefficient field. Rings
//! are shared between threads via `Arc<Ring>` (see
//! `~/project/docs/rust-bba-port-plan.md` §6.1); the type is `Send + Sync`
//! because it holds only immutable data.
//!
//! This bootstrap fixes two representation parameters:
//!
//! * **Ordering**: [`MonoOrder::DegRevLex`] only.
//! * **Bits per variable**: 8 bits. With 25 variables we use 200 bits
//!   (plus an 8-bit total-degree byte) which packs into 4 × u64 words.
//!   A variable exponent may therefore range from 0 to 255; the total
//!   degree is stored separately as a `u32` on each monomial, so
//!   degrees above 255 are still representable (they just can't be
//!   concentrated in a single variable).
//!
//! Future widths (16, 32 bits) will be added as follow-up work once the
//! bba driver is running against 8-bit workloads.

use crate::field::Field;
use crate::ordering::MonoOrder;

/// Bits used to store each variable's exponent in the packed monomial.
/// Fixed at 8 for this bootstrap.
pub const BITS_PER_VAR: u8 = 8;

/// Maximum number of variables supported by the 8-bit packing.
///
/// One 8-bit byte is reserved at the front of the packed representation
/// for total degree, leaving 31 bytes of a 256-bit (four-word)
/// exponent block for variables. The port plan aims at 25-variable
/// staging workloads, so 31 gives comfortable headroom.
pub const MAX_VARS: u32 = 31;

/// An immutable polynomial ring.
///
/// Construct via [`Ring::new`]. Share via `Arc<Ring>`. Never mutated
/// after construction; every method takes `&self`.
#[derive(Debug, Clone)]
pub struct Ring {
    /// Number of variables. `1 ≤ nvars ≤ MAX_VARS`.
    nvars: u32,
    /// Monomial ordering. Currently always [`MonoOrder::DegRevLex`].
    ordering: MonoOrder,
    /// Coefficient field Z/pZ.
    field: Field,
}

impl Ring {
    /// Construct a new ring.
    ///
    /// Returns `None` if `nvars` is out of range (`0` or `> MAX_VARS`)
    /// or if the caller passes an unsupported ordering. Today only
    /// `DegRevLex` is supported.
    pub fn new(nvars: u32, ordering: MonoOrder, field: Field) -> Option<Self> {
        if nvars == 0 || nvars > MAX_VARS {
            return None;
        }
        // Ordering is an exhaustive match; kept as `match` so future
        // variants must consciously opt in.
        match ordering {
            MonoOrder::DegRevLex => {}
        }
        Some(Self {
            nvars,
            ordering,
            field,
        })
    }

    /// Number of variables.
    #[inline]
    pub fn nvars(&self) -> u32 {
        self.nvars
    }

    /// Monomial ordering.
    #[inline]
    pub fn ordering(&self) -> MonoOrder {
        self.ordering
    }

    /// Coefficient field.
    #[inline]
    pub fn field(&self) -> &Field {
        &self.field
    }
}

impl PartialEq for Ring {
    fn eq(&self, other: &Self) -> bool {
        self.nvars == other.nvars && self.ordering == other.ordering && self.field == other.field
    }
}
impl Eq for Ring {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_valid_ring() {
        let f = Field::new(32003).unwrap();
        let r = Ring::new(5, MonoOrder::DegRevLex, f).unwrap();
        assert_eq!(r.nvars(), 5);
    }

    #[test]
    fn rejects_out_of_range_nvars() {
        let f = Field::new(5).unwrap();
        assert!(Ring::new(0, MonoOrder::DegRevLex, f).is_none());
        assert!(Ring::new(MAX_VARS + 1, MonoOrder::DegRevLex, f).is_none());
        assert!(Ring::new(MAX_VARS, MonoOrder::DegRevLex, f).is_some());
    }
}
