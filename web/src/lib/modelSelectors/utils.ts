import type { SelectDivider, SelectOption } from "@opal/components";
import { getModelIcon } from "@/lib/languageModels";
import {
  buildLlmOptions,
  groupLlmOptions,
  type ModelOptionProvider,
} from "@/lib/languageModels/options";

/** A model configuration id as an `InputSingleSelect` value; null is empty. */
export function toSelectValue(modelConfigurationId: number | null): string {
  return modelConfigurationId === null ? "" : String(modelConfigurationId);
}

/** The inverse of `toSelectValue`; empty and junk map to null. */
export function fromSelectValue(value: string): number | null {
  if (value === "") return null;
  const id = Number(value);
  return Number.isInteger(id) ? id : null;
}

/**
 * Every model configuration given, as Opal options: one titled divider per
 * provider (aggregators split per vendor, as the chat picker does), each
 * model a row with its icon. Nothing is filtered here; callers trim the
 * list first. A model without a configuration id cannot be chosen by id,
 * so it is left out.
 */
export function buildModelSelectOptions(
  providers: ModelOptionProvider[]
): SelectDivider[] {
  return groupLlmOptions(buildLlmOptions(providers, undefined, true))
    .map((group) => ({
      title: group.displayName,
      options: group.options.flatMap((option): SelectOption[] =>
        option.modelConfigurationId == null
          ? []
          : [
              {
                value: String(option.modelConfigurationId),
                title: option.displayName,
                icon: getModelIcon(option.provider, option.modelName),
              },
            ]
      ),
    }))
    .filter((divider) => divider.options.length > 0);
}
